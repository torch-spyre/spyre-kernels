// Top-k Log-softmax kernel in KTDP dialect
//
// Algorithm (two-pass log-softmax, accumulated in f32):
//   1. Find max logit across vocab (f32 accumulation)
//   2. Compute sum(exp(logit_f32 - max)) via block loop (f32)
//   3. lse = log(sum)
//   4. output[k] = topk_logit_f32[k] - max - lse
//
// Uses construct_indirect_access_tile with ind() to gather the top-k logit
// values on-chip: logits[row, topk_ids[row, k]] for k=0..7.
//
// Concrete sizes: 32 rows, vocab_size=4096, topk=8, BLOCK_SIZE=1024
// Grid: [32, 1] — one core per row

module {
  func.func @log_softmax_kernel(
      %logits: index,        // logits 32x4096xf16
      %topk_ids: index,      // topk_ids 32x8xi64
      %output: index,        // output 32x8xf16
      %vocab_size: index,    // 4096
      %topk: index,          // 8
      %BLOCK_SIZE: index     // 1024
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %f0_f32 = arith.constant 0.0 : f32
    %neg_inf_f32 = arith.constant 0xFF800000 : f32

    %logits_view = ktdp.construct_memory_view %logits, sizes: [32, 4096], strides: [4096, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 4095 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x4096xf16>

    %topk_ids_view = ktdp.construct_memory_view %topk_ids, sizes: [32, 8], strides: [8, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x8xi64>

    %output_view = ktdp.construct_memory_view %output, sizes: [32, 8], strides: [8, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x8xf16>

    scf.for %row = %core_id to %c32 step %c32 : index {

        // === Pass 1: Find max logit (f32) ===
        %max_init = tensor.splat %neg_inf_f32 : tensor<1x1024xf32>

        %max_block = scf.for %col = %c0 to %vocab_size step %BLOCK_SIZE
            iter_args(%running_max = %max_init) -> tensor<1x1024xf32> {

            %L_acc = ktdp.construct_access_tile %logits_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %logit_f16 = ktdp.load %L_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %logit_f32 = arith.extf %logit_f16 : tensor<1x1024xf16> to tensor<1x1024xf32>
            %new_max = arith.maximumf %running_max, %logit_f32 : tensor<1x1024xf32>

            scf.yield %new_max : tensor<1x1024xf32>
        }

        // Reduce 1x1024 → scalar max (f32)
        %max_reduce_init = tensor.splat %neg_inf_f32 : tensor<1xf32>
        %max_row = linalg.reduce { arith.maximumf }
            ins(%max_block : tensor<1x1024xf32>)
            outs(%max_reduce_init : tensor<1xf32>)
            dimensions = [1]

        %c0_ext = arith.constant 0 : index
        %max_scalar = tensor.extract %max_row[%c0_ext] : tensor<1xf32>

        // === Pass 2: sum(exp(logit - max)) in f32 ===
        %zero_block = arith.constant dense<0.0> : tensor<1x1024xf32>
        %max_splat = tensor.splat %max_scalar : tensor<1x1024xf32>

        %sum_block = scf.for %col = %c0 to %vocab_size step %BLOCK_SIZE
            iter_args(%running_sum = %zero_block) -> tensor<1x1024xf32> {

            %L_acc2 = ktdp.construct_access_tile %logits_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %logit_f16_2 = ktdp.load %L_acc2 : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %logit_f32_2 = arith.extf %logit_f16_2 : tensor<1x1024xf16> to tensor<1x1024xf32>

            %shifted = arith.subf %logit_f32_2, %max_splat : tensor<1x1024xf32>
            %exp_shifted = math.exp %shifted : tensor<1x1024xf32>

            %new_sum = arith.addf %running_sum, %exp_shifted : tensor<1x1024xf32>

            scf.yield %new_sum : tensor<1x1024xf32>
        }

        // Reduce 1x1024 → scalar sum (f32)
        %sum_reduce_init = tensor.splat %f0_f32 : tensor<1xf32>
        %sum_row = linalg.reduce { arith.addf }
            ins(%sum_block : tensor<1x1024xf32>)
            outs(%sum_reduce_init : tensor<1xf32>)
            dimensions = [1]

        %sum_scalar = tensor.extract %sum_row[%c0_ext] : tensor<1xf32>

        // lse = log(sum) (f32)
        %lse_tile = tensor.splat %sum_scalar : tensor<1xf32>
        %log_tile = math.log %lse_tile : tensor<1xf32>
        %lse_scalar = tensor.extract %log_tile[%c0_ext] : tensor<1xf32>

        // === Gather top-k logits via ind() and compute output ===
        %topk_acc = ktdp.construct_indirect_access_tile
            intermediate_variables (%d0, %d1)
            %logits_view[(%row + %d0), ind(%topk_ids_view[%row + %d0, %d1])] {
              variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
              variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<32x4096xf16>, memref<32x8xi64> -> !ktdp.access_tile<1x8xindex>

        %topk_f16 = ktdp.load %topk_acc : !ktdp.access_tile<1x8xindex> -> tensor<1x8xf16>
        %topk_f32 = arith.extf %topk_f16 : tensor<1x8xf16> to tensor<1x8xf32>

        %max_splat8 = tensor.splat %max_scalar : tensor<1x8xf32>
        %lse_splat8 = tensor.splat %lse_scalar : tensor<1x8xf32>

        %shifted_topk = arith.subf %topk_f32, %max_splat8 : tensor<1x8xf32>
        %result_f32 = arith.subf %shifted_topk, %lse_splat8 : tensor<1x8xf32>
        %result = arith.truncf %result_f32 : tensor<1x8xf32> to tensor<1x8xf16>

        %out_acc = ktdp.construct_access_tile %output_view[%row, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x8xf16> -> !ktdp.access_tile<1x8xindex>

        ktdp.store %result, %out_acc : tensor<1x8xf16>, !ktdp.access_tile<1x8xindex>

        scf.yield
    }
    return
  }
}
