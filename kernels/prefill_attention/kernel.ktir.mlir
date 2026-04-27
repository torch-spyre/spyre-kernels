// Prefill Attention (SDPA) kernel in KTDP dialect
//
// Multi-head SDPA with causal masking.
// Q, K, V share the same sequence length (single sequence, no batching).
//
// Algorithm (per head):
//   QK = Q[h] @ K[h]^T * scale        [seq, seq]
//   QK = QK + causal_mask              (mask upper triangle with -1e8)
//   P = softmax(QK, dim=-1)            [seq, seq]
//   O = P @ V[h]                       [seq, hd]
//
// The block-pointer Triton kernel uses online softmax (flash attention tiling)
// for memory efficiency. At seq_len=16, materializing the full matrix is
// numerically equivalent (no tiling needed). The causal mask and softmax
// logic match exactly.
//
// Concrete: seq_len=16, num_heads=4, head_dim=64
// Q/K/V: [16, 256] (seq_len x num_heads*head_dim), Output: [16, 256]
// Grid: [4, 1] — one core per head

module {
  func.func @prefill_attention_kernel(
      %q_ptr: index,       // Q: [16, 256] xf16
      %k_ptr: index,       // K: [16, 256] xf16
      %v_ptr: index,       // V: [16, 256] xf16
      %output_ptr: index,  // Output: [16, 256] xf16
      %causal_mask_ptr: index,  // Causal mask: [16, 16] xf16 (0 or -1e8)
      %num_heads: index    // 4
  ) attributes {grid = [4, 1]} {
    %core_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %zero_f16 = arith.constant 0.0 : f16
    %scale = arith.constant 1.25e-01 : f16  // 1/sqrt(64) = 0.125
    %neg_inf = arith.constant 0xFC00 : f16
    %mask_val = arith.constant -1.0e+4 : f16  // -10000, fits in f16 range

    // Q: [16, 256]
    %q_view = ktdp.construct_memory_view %q_ptr, sizes: [16, 256], strides: [256, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<16x256xf16>

    // K: [16, 256]
    %k_view = ktdp.construct_memory_view %k_ptr, sizes: [16, 256], strides: [256, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<16x256xf16>

    // V: [16, 256]
    %v_view = ktdp.construct_memory_view %v_ptr, sizes: [16, 256], strides: [256, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<16x256xf16>

    // Output: [16, 256]
    %out_view = ktdp.construct_memory_view %output_ptr, sizes: [16, 256], strides: [256, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 255 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<16x256xf16>

    // Causal mask: [16, 16] (pre-computed: 0 on lower triangle, -1e8 on upper)
    %mask_view = ktdp.construct_memory_view %causal_mask_ptr, sizes: [16, 16], strides: [16, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<16x16xf16>

    // Each core handles one head
    scf.for %head = %core_id to %c4 step %c4 : index {

        %col = arith.muli %head, %c64 : index

        // Load Q head: [16, 64]
        %q_acc = ktdp.construct_access_tile %q_view[%c0, %col] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<16x256xf16> -> !ktdp.access_tile<16x64xindex>

        %q = ktdp.load %q_acc : !ktdp.access_tile<16x64xindex> -> tensor<16x64xf16>

        // Load K head: [16, 64]
        %k_acc = ktdp.construct_access_tile %k_view[%c0, %col] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<16x256xf16> -> !ktdp.access_tile<16x64xindex>

        %k = ktdp.load %k_acc : !ktdp.access_tile<16x64xindex> -> tensor<16x64xf16>

        // QK = Q @ K^T: [16, 64] @ [64, 16] -> [16, 16]
        %k_t_init = tensor.empty() : tensor<64x16xf16>
        %k_t = linalg.transpose ins(%k : tensor<16x64xf16>)
                                outs(%k_t_init : tensor<64x16xf16>)
                                permutation = [1, 0]

        %qk_init = tensor.empty() : tensor<16x16xf16>
        %qk = linalg.matmul ins(%q, %k_t : tensor<16x64xf16>, tensor<64x16xf16>)
                            outs(%qk_init : tensor<16x16xf16>) -> tensor<16x16xf16>

        // Scale: QK * scale
        %scale_splat = tensor.splat %scale : tensor<16x16xf16>
        %qk_scaled = arith.mulf %qk, %scale_splat : tensor<16x16xf16>

        // Load pre-computed causal mask: [16, 16]
        %mask_acc = ktdp.construct_access_tile %mask_view[%c0, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 15 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<16x16xf16> -> !ktdp.access_tile<16x16xindex>

        %causal_mask = ktdp.load %mask_acc : !ktdp.access_tile<16x16xindex> -> tensor<16x16xf16>

        %qk_masked = arith.addf %qk_scaled, %causal_mask : tensor<16x16xf16>

        // Softmax: row-wise max
        %mi_init = tensor.empty() : tensor<16xf16>
        %mi_neginf = linalg.fill ins(%neg_inf : f16) outs(%mi_init : tensor<16xf16>) -> tensor<16xf16>
        %mi = linalg.reduce { arith.maxf }
                ins(%qk_masked : tensor<16x16xf16>)
                outs(%mi_neginf : tensor<16xf16>)
                dimensions = [1]

        // P = exp(QK_masked - max)
        %mi_bc_init = tensor.empty() : tensor<16x16xf16>
        %mi_bc = linalg.broadcast ins(%mi : tensor<16xf16>)
                                  outs(%mi_bc_init : tensor<16x16xf16>)
                                  dimensions = [1]
        %qk_shifted = arith.subf %qk_masked, %mi_bc : tensor<16x16xf16>
        %p = math.exp %qk_shifted : tensor<16x16xf16>

        // Sum exp: l_i = sum(P, dim=1)
        %li_init = tensor.empty() : tensor<16xf16>
        %li_zeros = linalg.fill ins(%zero_f16 : f16) outs(%li_init : tensor<16xf16>) -> tensor<16xf16>
        %li = linalg.reduce { arith.addf }
                ins(%p : tensor<16x16xf16>)
                outs(%li_zeros : tensor<16xf16>)
                dimensions = [1]

        // P_norm = P / l_i
        %li_bc_init = tensor.empty() : tensor<16x16xf16>
        %li_bc = linalg.broadcast ins(%li : tensor<16xf16>)
                                  outs(%li_bc_init : tensor<16x16xf16>)
                                  dimensions = [1]
        %p_norm = arith.divf %p, %li_bc : tensor<16x16xf16>

        // Load V head: [16, 64]
        %v_acc = ktdp.construct_access_tile %v_view[%c0, %col] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<16x256xf16> -> !ktdp.access_tile<16x64xindex>

        %v = ktdp.load %v_acc : !ktdp.access_tile<16x64xindex> -> tensor<16x64xf16>

        // Output = P_norm @ V: [16, 16] @ [16, 64] -> [16, 64]
        %acc_init = tensor.empty() : tensor<16x64xf16>
        %acc_zeros = linalg.fill ins(%zero_f16 : f16) outs(%acc_init : tensor<16x64xf16>) -> tensor<16x64xf16>
        %acc = linalg.matmul ins(%p_norm, %v : tensor<16x16xf16>, tensor<16x64xf16>)
                             outs(%acc_zeros : tensor<16x64xf16>) -> tensor<16x64xf16>

        // Store output head: [16, 64]
        %out_acc = ktdp.construct_access_tile %out_view[%c0, %col] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<16x256xf16> -> !ktdp.access_tile<16x64xindex>

        ktdp.store %acc, %out_acc : tensor<16x64xf16>, !ktdp.access_tile<16x64xindex>

        scf.yield
    }
    return
  }
}
