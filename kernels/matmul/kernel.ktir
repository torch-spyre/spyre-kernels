// Matrix multiplication kernel in KTDP dialect
//
// Algorithm: C[m, n] = sum_k(A[m, k] * B[k, n])
//
// Accumulation is done in f32 to match the block-pointer Triton kernel.
// Output is truncated to f16.
//
// Tiling strategy:
//   - Grid of 32 cores, 1D
//   - Each core processes output tiles of size [BLOCK_M, BLOCK_N]
//   - Tiles are assigned to cores in round-robin fashion across the M dimension
//   - Inner loop tiles over K in steps of BLOCK_K
//
// Concrete sizes: A [128, 128] f16, B [128, 128] f16, C [128, 128] f16
// BLOCK_M=32, BLOCK_N=128, BLOCK_K=32
// Grid: [4, 1] — 4 cores (128/32 M-tiles), each core handles one M-tile
//                with full N (single BLOCK_N=128 covers N)

module {
  func.func @matmul_kernel(
      %A: index,       // A: [128, 128] xf16
      %B: index,       // B: [128, 128] xf16
      %C: index,       // C: [128, 128] xf16
      %M: index,       // 128
      %N: index,       // 128
      %K: index        // 128
  ) attributes {grid = [4, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index    // BLOCK_M and BLOCK_K
    %c128 = arith.constant 128 : index  // BLOCK_N (= N)
    %zero_f32 = arith.constant 0.0 : f32

    // A view: [128, 128] row-major
    %A_view = ktdp.construct_memory_view %A, sizes: [128, 128], strides: [128, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x128xf16>

    // B view: [128, 128] row-major
    %B_view = ktdp.construct_memory_view %B, sizes: [128, 128], strides: [128, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x128xf16>

    // C view: [128, 128] row-major
    %C_view = ktdp.construct_memory_view %C, sizes: [128, 128], strides: [128, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x128xf16>

    // Each core handles M-tiles: core 0 -> rows [0:32], core 1 -> [32:64], etc.
    %m_start = arith.muli %core_id, %c32 : index

    scf.for %m_tile = %m_start to %M step %c128 : index {

        // Initialize accumulator: [BLOCK_M, BLOCK_N] = [32, 128] in f32
        %acc_init = tensor.splat %zero_f32 : tensor<32x128xf32>

        // Loop over K in blocks of BLOCK_K=32
        %acc_final = scf.for %k = %c0 to %K step %c32
            iter_args(%acc = %acc_init) -> tensor<32x128xf32> {

            // Load A tile: [32, 32] from A[m_tile:m_tile+32, k:k+32]
            %A_acc = ktdp.construct_access_tile %A_view[%m_tile, %k] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x128xf16> -> !ktdp.access_tile<32x32xindex>

            %a_f16 = ktdp.load %A_acc : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf16>

            // Load B tile: [32, 128] from B[k:k+32, 0:128]
            %B_acc = ktdp.construct_access_tile %B_view[%k, %c0] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x128xf16> -> !ktdp.access_tile<32x128xindex>

            %b_f16 = ktdp.load %B_acc : !ktdp.access_tile<32x128xindex> -> tensor<32x128xf16>

            // Upcast to f32 for accumulation
            %a_f32 = arith.extf %a_f16 : tensor<32x32xf16> to tensor<32x32xf32>
            %b_f32 = arith.extf %b_f16 : tensor<32x128xf16> to tensor<32x128xf32>

            // Matmul: [32, 32] @ [32, 128] -> [32, 128], accumulated into acc
            %prod = linalg.matmul ins(%a_f32, %b_f32 : tensor<32x32xf32>, tensor<32x128xf32>)
                                  outs(%acc : tensor<32x128xf32>) -> tensor<32x128xf32>

            scf.yield %prod : tensor<32x128xf32>
        }

        // Truncate f32 accumulator to f16 for output
        %c_f16 = arith.truncf %acc_final : tensor<32x128xf32> to tensor<32x128xf16>

        // Store C tile: [32, 128] to C[m_tile:m_tile+32, 0:128]
        %C_acc = ktdp.construct_access_tile %C_view[%m_tile, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<128x128xf16> -> !ktdp.access_tile<32x128xindex>

        ktdp.store %c_f16, %C_acc : tensor<32x128xf16>, !ktdp.access_tile<32x128xindex>

        scf.yield
    }
    return
  }
}
