// RMSNorm forward kernel in KTDP dialect
//
// Algorithm: y[row, col] = x[row, col] / sqrt(mean(x[row, :]^2) + eps) * weight[col]
//
// Each core processes rows core_id, core_id+32, core_id+64, ...
// Within each row, columns are tiled in blocks of 1024.
//
// Accumulation is done in f32 to match the block-pointer Triton kernel,
// which upcasts to float32 before squaring and accumulating.
//
// Concrete sizes: 32 rows × 4096 columns, weight is 4096, eps is a scalar f16 arg.
// Grid: [32, 1] — 32 cores, 1D.

module {
  func.func @rms_norm_fwd(
      %X: index,       // input  32x4096xf16
      %W: index,       // weight 4096xf16
      %Y: index,       // output 32x4096xf16
      %N: index,       // number of columns: 4096
      %eps: f16,
      %BLOCK_SIZE: index  // 1024
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c32_rows = arith.constant 32 : index
    %f1_f32 = arith.constant 1.0 : f32

    %X_view = ktdp.construct_memory_view %X, sizes: [32, 4096], strides: [4096, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 4095 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x4096xf16>

    %W_view = ktdp.construct_memory_view %W, sizes: [4096], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 4095 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096xf16>

    %Y_view = ktdp.construct_memory_view %Y, sizes: [32, 4096], strides: [4096, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 4095 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x4096xf16>

    scf.for %row = %core_id to %c32_rows step %c32 : index {

        // ============================================================
        // Pass 1: sum of squares (accumulated in f32)
        // ============================================================
        %zero_block_f32 = arith.constant dense<0.0> : tensor<1x1024xf32>

        %sum_sq_block = scf.for %col = %c0 to %N step %BLOCK_SIZE
            iter_args(%acc = %zero_block_f32) -> tensor<1x1024xf32> {

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %x_f16 = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            // Upcast to f32
            %x_f32 = arith.extf %x_f16 : tensor<1x1024xf16> to tensor<1x1024xf32>

            // x^2
            %x2 = arith.mulf %x_f32, %x_f32 : tensor<1x1024xf32>

            // accumulate
            %next_acc = arith.addf %acc, %x2 : tensor<1x1024xf32>

            scf.yield %next_acc : tensor<1x1024xf32>
        }

        // Reduce 1x1024 -> 1 (sum across columns) in f32
        %zero_scalar_f32 = arith.constant 0.0 : f32
        %sum_init_f32 = tensor.splat %zero_scalar_f32 : tensor<1xf32>
        %sum_sq = linalg.reduce { arith.addf }
            ins(%sum_sq_block : tensor<1x1024xf32>)
            outs(%sum_init_f32 : tensor<1xf32>)
            dimensions = [1]

        // mean_sq = sum_sq / N  (f32)
        %N_i32 = arith.index_cast %N : index to i32
        %N_f32 = arith.sitofp %N_i32 : i32 to f32
        %N_tensor_f32 = tensor.splat %N_f32 : tensor<1xf32>
        %mean_sq = arith.divf %sum_sq, %N_tensor_f32 : tensor<1xf32>

        // rms = sqrt(mean_sq + eps)  (f32)
        %eps_f32 = arith.extf %eps : f16 to f32
        %eps_tensor_f32 = tensor.splat %eps_f32 : tensor<1xf32>
        %mean_sq_eps = arith.addf %mean_sq, %eps_tensor_f32 : tensor<1xf32>
        %rms = math.sqrt %mean_sq_eps : tensor<1xf32>

        // inv_rms = 1.0 / rms  (f32)
        %f1_tensor = tensor.splat %f1_f32 : tensor<1xf32>
        %inv_rms = arith.divf %f1_tensor, %rms : tensor<1xf32>

        // Broadcast inv_rms to 1x1024 for element-wise multiply
        %c0_extract = arith.constant 0 : index
        %inv_rms_scalar = tensor.extract %inv_rms[%c0_extract] : tensor<1xf32>
        %inv_rms_block = tensor.splat %inv_rms_scalar : tensor<1x1024xf32>

        // ============================================================
        // Pass 2: normalize and apply weight (compute in f32, store f16)
        // ============================================================
        scf.for %col = %c0 to %N step %BLOCK_SIZE {

            %X_acc2 = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %W_acc = ktdp.construct_access_tile %W_view[%col] {
                access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0) -> (d0)>
            } : memref<4096xf16> -> !ktdp.access_tile<1024xindex>

            %Y_acc = ktdp.construct_access_tile %Y_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %x2_f16 = ktdp.load %X_acc2 : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %w_f16 = ktdp.load %W_acc : !ktdp.access_tile<1024xindex> -> tensor<1024xf16>

            // Upcast to f32
            %x2_f32 = arith.extf %x2_f16 : tensor<1x1024xf16> to tensor<1x1024xf32>
            %w_1d_f32 = arith.extf %w_f16 : tensor<1024xf16> to tensor<1024xf32>
            %w_block_f32 = tensor.expand_shape %w_1d_f32 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>

            // Normalize: x * inv_rms * weight  (all f32)
            %normed = arith.mulf %x2_f32, %inv_rms_block : tensor<1x1024xf32>
            %y_f32 = arith.mulf %normed, %w_block_f32 : tensor<1x1024xf32>

            // Downcast to f16 for storage
            %y_f16 = arith.truncf %y_f32 : tensor<1x1024xf32> to tensor<1x1024xf16>

            ktdp.store %y_f16, %Y_acc : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

            scf.yield
        }
        scf.yield
    }
    return
  }
}
