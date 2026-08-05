// Embedding forward kernel in KTDP dialect
//
// Algorithm: For each token, gather the full row from the embedding table:
//            output[i, :] = embeddings[indices[i], :]
//
// Uses construct_indirect_access_tile with ind() to perform the row gather
// on-chip: embeddings[indices[i], :]. This is the same pattern as the
// ranks kernel's scalar gather, but along the embedding_dim axis.
//
// Tiling strategy:
//   - Grid of 32 cores, 1D
//   - Each core handles one token: core i processes row i
//   - Full embedding_dim (1024) fits in one tile per token
//
// Concrete sizes: n_tokens=32, vocab_size=4096, embedding_dim=1024
// Grid: [32, 1] — 32 cores, one core per token

module {
  func.func @embedding_kernel(
      %embeddings: index,  // embeddings vocab_sizexembedding_dimxf16
      %indices: index,     // indices n_tokensxi64
      %output: index       // output n_tokensxembedding_dimxf16
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index

    %embeddings_view = ktdp.construct_memory_view %embeddings, sizes: [4096, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096x1024xf16>

    %indices_view = ktdp.construct_memory_view %indices, sizes: [32], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32xi64>

    %output_view = ktdp.construct_memory_view %output, sizes: [32, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x1024xf16>

    // Indirect gather: embeddings[indices[core_id], :]
    %emb_acc = ktdp.construct_indirect_access_tile
        intermediate_variables (%d0, %d1)
        %embeddings_view[ind(%indices_view[%core_id + %d0]), %d1] {
          variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
          variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<4096x1024xf16>, memref<32xi64> -> !ktdp.access_tile<1x1024xindex>

    %emb_tile = ktdp.load %emb_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

    // Store to output[core_id, :]
    %out_acc = ktdp.construct_access_tile %output_view[%core_id, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x1024xf16> -> !ktdp.access_tile<1x1024xindex>

    ktdp.store %emb_tile, %out_acc : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

    return
  }
}
