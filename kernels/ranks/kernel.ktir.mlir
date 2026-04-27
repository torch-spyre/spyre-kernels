// Ranks kernel in KTDP dialect
//
// Algorithm: For each row, count how many logit values are >= the reference
//            logit (at position token_id). Output is a count per row.
//
// Interface difference from block-pointer Triton kernel:
// The block-ptr kernel takes token_ids and does a data-dependent gather:
//   ref_logit = logits[row, token_ids[row]]
// KTDP cannot express data-dependent gathers, so we pre-extract reference
// logits on the host: ref_logits[i] = logits[i, token_ids[i]]
// This is semantically equivalent — the gather is just hoisted to the caller.
//
// Concrete sizes: 32 rows, vocab_size=4096, BLOCK_SIZE=1024
// Grid: [32, 1]

module {
  func.func @ranks_kernel(
      %logits: index,      // logits 32x4096xf16
      %ref_logits: index,  // ref_logits 32xf16 (pre-extracted reference values)
      %output: index,      // output 32xf16 (counts as f16)
      %vocab_size: index,  // 4096
      %BLOCK_SIZE: index   // 1024
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %f0 = arith.constant 0.0 : f16
    %f1 = arith.constant 1.0 : f16

    %logits_view = ktdp.construct_memory_view %logits, sizes: [32, 4096], strides: [4096, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 4095 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x4096xf16>

    %ref_view = ktdp.construct_memory_view %ref_logits, sizes: [32], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32xf16>

    %output_view = ktdp.construct_memory_view %output, sizes: [32], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32xf16>

    scf.for %row = %core_id to %c32 step %c32 : index {

        // Load the reference logit for this row
        %ref_acc = ktdp.construct_access_tile %ref_view[%row] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<32xf16> -> !ktdp.access_tile<1xindex>

        %ref_tile = ktdp.load %ref_acc : !ktdp.access_tile<1xindex> -> tensor<1xf16>
        %c0_ext = arith.constant 0 : index
        %ref_scalar = tensor.extract %ref_tile[%c0_ext] : tensor<1xf16>
        %ref_block = tensor.splat %ref_scalar : tensor<1x1024xf16>

        // Accumulate count: loop over vocab blocks
        %zero_block = arith.constant dense<0.0> : tensor<1x1024xf16>

        %count_block = scf.for %col = %c0 to %vocab_size step %BLOCK_SIZE
            iter_args(%acc = %zero_block) -> tensor<1x1024xf16> {

            %L_acc = ktdp.construct_access_tile %logits_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %logit_block = ktdp.load %L_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            // Compare: logits >= ref (using cmpi sge which the interpreter
            // evaluates as numpy >= on the underlying f16 data)
            %cmp = arith.cmpi sge, %logit_block, %ref_block : tensor<1x1024xf16>

            // cmp is a boolean tile; select 1.0 where true, 0.0 where false
            %ones = tensor.splat %f1 : tensor<1x1024xf16>
            %zeros = tensor.splat %f0 : tensor<1x1024xf16>
            %indicator = arith.select %cmp, %ones, %zeros : tensor<1x1024xf16>

            // Accumulate
            %next_acc = arith.addf %acc, %indicator : tensor<1x1024xf16>

            scf.yield %next_acc : tensor<1x1024xf16>
        }

        // Reduce 1x1024 → scalar (sum of indicators = count)
        %sum_init = tensor.splat %f0 : tensor<1xf16>
        %count = linalg.reduce { arith.addf }
            ins(%count_block : tensor<1x1024xf16>)
            outs(%sum_init : tensor<1xf16>)
            dimensions = [1]

        // Store the count
        %out_acc = ktdp.construct_access_tile %output_view[%row] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<32xf16> -> !ktdp.access_tile<1xindex>

        ktdp.store %count, %out_acc : tensor<1xf16>, !ktdp.access_tile<1xindex>

        scf.yield
    }
    return
  }
}
