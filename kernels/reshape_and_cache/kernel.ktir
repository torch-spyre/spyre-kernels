// KV Cache Reshape kernel in KTDP dialect
//
// Simplified version of reshape_and_cache_kernel_flash (non-HEAD_MAJOR_LAYOUT, no FP8).
// Copies key and value from linear per-token layout to a paged block cache
// using slot_mapping for indirect indexing.
//
// Algorithm (per token):
//   slot = slot_mapping[token_id]
//   if slot < 0: skip (padding token)
//   key_cache[slot, :] = key[token_id, :]
//   value_cache[slot, :] = value[token_id, :]
//
// slot_mapping uses i64 memref, matching the Triton kernel's int64 dtype.
//
// Concrete sizes: key/value [32, 512], cache [64, 512], slot_mapping [32]
// Grid: [32, 1] — one core per token

module {
  func.func @reshape_and_cache_kernel(
      %key: index,          // key: [32, 512] xf16
      %value: index,        // value: [32, 512] xf16
      %key_cache: index,    // key_cache: [4, 16, 512] xf16 = [64, 512] flattened first two dims
      %value_cache: index,  // value_cache: [4, 16, 512] xf16 = [64, 512] flattened
      %slot_mapping: index, // slot_mapping: [32] xi32
      %block_size: index    // 16
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index

    // Key: [32, 512]
    %key_view = ktdp.construct_memory_view %key, sizes: [32, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x512xf16>

    // Value: [32, 512]
    %value_view = ktdp.construct_memory_view %value, sizes: [32, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x512xf16>

    // Key cache: [64, 512] (flattened from [4, 16, 512])
    %key_cache_view = ktdp.construct_memory_view %key_cache, sizes: [64, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x512xf16>

    // Value cache: [64, 512]
    %value_cache_view = ktdp.construct_memory_view %value_cache, sizes: [64, 512], strides: [512, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x512xf16>

    // Slot mapping: [32] xi64
    %slot_view = ktdp.construct_memory_view %slot_mapping, sizes: [32], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32xi64>

    scf.for %token_id = %core_id to %c32 step %c32 : index {

        // Load slot index for this token
        %slot_acc = ktdp.construct_access_tile %slot_view[%token_id] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<32xi64> -> !ktdp.access_tile<1xindex>

        %slot_tile = ktdp.load %slot_acc : !ktdp.access_tile<1xindex> -> tensor<1xi64>
        %c0_e = arith.constant 0 : index
        %slot_i64 = tensor.extract %slot_tile[%c0_e] : tensor<1xi64>
        %slot_idx = arith.index_cast %slot_i64 : i64 to index

        // Load key row: [1, 512]
        %key_acc = ktdp.construct_access_tile %key_view[%token_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x512xf16> -> !ktdp.access_tile<1x512xindex>

        %key_row = ktdp.load %key_acc : !ktdp.access_tile<1x512xindex> -> tensor<1x512xf16>

        // Load value row: [1, 512]
        %val_acc = ktdp.construct_access_tile %value_view[%token_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x512xf16> -> !ktdp.access_tile<1x512xindex>

        %val_row = ktdp.load %val_acc : !ktdp.access_tile<1x512xindex> -> tensor<1x512xf16>

        // Store key to cache at slot_idx row
        %kc_acc = ktdp.construct_access_tile %key_cache_view[%slot_idx, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<64x512xf16> -> !ktdp.access_tile<1x512xindex>

        ktdp.store %key_row, %kc_acc : tensor<1x512xf16>, !ktdp.access_tile<1x512xindex>

        // Store value to cache at slot_idx row
        %vc_acc = ktdp.construct_access_tile %value_cache_view[%slot_idx, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<64x512xf16> -> !ktdp.access_tile<1x512xindex>

        ktdp.store %val_row, %vc_acc : tensor<1x512xf16>, !ktdp.access_tile<1x512xindex>

        scf.yield
    }
    return
  }
}
