// MRoPE (Multi-dimensional Rotary Position Embedding) kernel in KTDP dialect
//
// Simplified version: applies rotary embedding to both Q and K tensors in-place.
// cos and sin are pre-merged from 3 positional dimensions (the T/H/W section
// masking and 3-source gather is done on the host, matching the block-pointer
// kernel where those loads remain as raw pointers).
//
// Algorithm (per token, per head):
//   x1 = q[token, head, 0:hd/2]     (left half)
//   x2 = q[token, head, hd/2:hd]    (right half)
//   new_x1 = x1 * cos - x2 * sin
//   new_x2 = x2 * cos + x1 * sin
//   (same for k)
//
// Concrete sizes: num_tokens=32, num_q_heads=8, num_kv_heads=8, head_dim=64
// Q is flattened: [32, 512] (num_tokens x num_q_heads*head_dim)
// K is flattened: [32, 512] (num_tokens x num_kv_heads*head_dim)
// cos, sin: [32, 32] (num_tokens x head_dim/2)
// Grid: [32, 1] — one core per token

module {
  func.func @mrope_kernel(
      %q: index,          // Q: [32, 512] xf16 (num_tokens x num_q_heads*head_dim)
      %k: index,          // K: [32, 512] xf16 (num_tokens x num_kv_heads*head_dim)
      %cos_ptr: index,    // cos: [32, 32] xf16 (num_tokens x head_dim/2)
      %sin_ptr: index,    // sin: [32, 32] xf16 (num_tokens x head_dim/2)
      %num_q_heads: index,  // 8
      %num_kv_heads: index  // 8
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index

    // Q: [32, 512]
    %q_view = ktdp.construct_memory_view %q, sizes: [32, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x512xf16>

    // K: [32, 512]
    %k_view = ktdp.construct_memory_view %k, sizes: [32, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x512xf16>

    // cos: [32, 32]
    %cos_view = ktdp.construct_memory_view %cos_ptr, sizes: [32, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x32xf16>

    // sin: [32, 32]
    %sin_view = ktdp.construct_memory_view %sin_ptr, sizes: [32, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x32xf16>

    scf.for %token_id = %core_id to %c32 step %c32 : index {

        // Load cos and sin for this token: [1, 32]
        %cos_acc = ktdp.construct_access_tile %cos_view[%token_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x32xf16> -> !ktdp.access_tile<1x32xindex>

        %cos_tile = ktdp.load %cos_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

        %sin_acc = ktdp.construct_access_tile %sin_view[%token_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x32xf16> -> !ktdp.access_tile<1x32xindex>

        %sin_tile = ktdp.load %sin_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

        // ─── Apply rotary to Q heads ───
        scf.for %head = %c0 to %num_q_heads step %c1 : index {

            %col_offset = arith.muli %head, %c64 : index

            // Load left half: q[token, head*64 : head*64+32]
            %q_x1_acc = ktdp.construct_access_tile %q_view[%token_id, %col_offset] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>

            %q_x1 = ktdp.load %q_x1_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

            // Load right half: q[token, head*64+32 : head*64+64]
            %c32_idx = arith.constant 32 : index
            %col_right = arith.addi %col_offset, %c32_idx : index

            %q_x2_acc = ktdp.construct_access_tile %q_view[%token_id, %col_right] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>

            %q_x2 = ktdp.load %q_x2_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

            // new_x1 = x1 * cos - x2 * sin
            %q_x1_cos = arith.mulf %q_x1, %cos_tile : tensor<1x32xf16>
            %q_x2_sin = arith.mulf %q_x2, %sin_tile : tensor<1x32xf16>
            %q_new_x1 = arith.subf %q_x1_cos, %q_x2_sin : tensor<1x32xf16>

            // new_x2 = x2 * cos + x1 * sin
            %q_x2_cos = arith.mulf %q_x2, %cos_tile : tensor<1x32xf16>
            %q_x1_sin = arith.mulf %q_x1, %sin_tile : tensor<1x32xf16>
            %q_new_x2 = arith.addf %q_x2_cos, %q_x1_sin : tensor<1x32xf16>

            // Store results back to Q
            %q_out_x1 = ktdp.construct_access_tile %q_view[%token_id, %col_offset] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>
            ktdp.store %q_new_x1, %q_out_x1 : tensor<1x32xf16>, !ktdp.access_tile<1x32xindex>

            %q_out_x2 = ktdp.construct_access_tile %q_view[%token_id, %col_right] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>
            ktdp.store %q_new_x2, %q_out_x2 : tensor<1x32xf16>, !ktdp.access_tile<1x32xindex>

            scf.yield
        }

        // ─── Apply rotary to K heads ───
        scf.for %head = %c0 to %num_kv_heads step %c1 : index {

            %col_offset_k = arith.muli %head, %c64 : index

            %k_x1_acc = ktdp.construct_access_tile %k_view[%token_id, %col_offset_k] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>

            %k_x1 = ktdp.load %k_x1_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

            %c32_k = arith.constant 32 : index
            %col_right_k = arith.addi %col_offset_k, %c32_k : index

            %k_x2_acc = ktdp.construct_access_tile %k_view[%token_id, %col_right_k] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>

            %k_x2 = ktdp.load %k_x2_acc : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf16>

            %k_x1_cos = arith.mulf %k_x1, %cos_tile : tensor<1x32xf16>
            %k_x2_sin = arith.mulf %k_x2, %sin_tile : tensor<1x32xf16>
            %k_new_x1 = arith.subf %k_x1_cos, %k_x2_sin : tensor<1x32xf16>

            %k_x2_cos = arith.mulf %k_x2, %cos_tile : tensor<1x32xf16>
            %k_x1_sin = arith.mulf %k_x1, %sin_tile : tensor<1x32xf16>
            %k_new_x2 = arith.addf %k_x2_cos, %k_x1_sin : tensor<1x32xf16>

            %k_out_x1 = ktdp.construct_access_tile %k_view[%token_id, %col_offset_k] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>
            ktdp.store %k_new_x1, %k_out_x1 : tensor<1x32xf16>, !ktdp.access_tile<1x32xindex>

            %k_out_x2 = ktdp.construct_access_tile %k_view[%token_id, %col_right_k] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x512xf16> -> !ktdp.access_tile<1x32xindex>
            ktdp.store %k_new_x2, %k_out_x2 : tensor<1x32xf16>, !ktdp.access_tile<1x32xindex>

            scf.yield
        }
        scf.yield
    }
    return
  }
}
