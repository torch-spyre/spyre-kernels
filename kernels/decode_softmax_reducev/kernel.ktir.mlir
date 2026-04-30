// Decode softmax + reduceV (stage 2) kernel in KTDP dialect
//
// Two-pass approach (equivalent to online softmax for pre-computed split data):
//   Pass 1: Find max logit and compute log-sum-exp across splits
//   Pass 2: Compute weighted sum of V vectors using softmax weights
//
// This avoids multi-result scf.for which the KTIR interpreter doesn't support.
// Numerically equivalent to the online softmax in the block-pointer kernel.
//
// Concrete sizes: batch=4, heads=8, NUM_KV_SPLITS=4, Lv=64
// Grid: [32, 1] — one core per (batch, head) pair

module {
  func.func @decode_softmax_reducev_kernel(
      %mid_o: index,       // Mid_O flattened: [128, 65] xf16 (V concat logit)
      %lse_out: index,     // lse: [32] xf16
      %output: index,      // output: [32, 64] xf16
      %num_splits: index   // 4
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %f0 = arith.constant 0.0 : f16
    %neg_inf = arith.constant 0xFC00 : f16

    %mid_o_view = ktdp.construct_memory_view %mid_o, sizes: [128, 65], strides: [65, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 64 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x65xf16>

    %output_view = ktdp.construct_memory_view %output, sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %lse_view = ktdp.construct_memory_view %lse_out, sizes: [32], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32xf16>

    scf.for %pair_id = %core_id to %c32 step %c32 : index {

        %base_row = arith.muli %pair_id, %c4 : index
        %c0_e = arith.constant 0 : index

        // ═══ Pass 1: Find max logit across splits ═══
        %max_init = tensor.splat %neg_inf : tensor<1xf16>

        %max_logit = scf.for %split = %c0 to %num_splits step %c1
            iter_args(%running_max = %max_init) -> tensor<1xf16> {

            %row = arith.addi %base_row, %split : index

            %logit_acc = ktdp.construct_access_tile %mid_o_view[%row, %c64] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 0 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x1xindex>

            %logit_tile = ktdp.load %logit_acc : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>
            %tlogic = tensor.extract %logit_tile[%c0_e, %c0_e] : tensor<1x1xf16>
            %tlogic_1d = tensor.splat %tlogic : tensor<1xf16>

            %new_max = arith.maximumf %running_max, %tlogic_1d : tensor<1xf16>
            scf.yield %new_max : tensor<1xf16>
        }

        %max_scalar = tensor.extract %max_logit[%c0_e] : tensor<1xf16>

        // ═══ Pass 1b: Compute sum(exp(logit - max)) ═══
        %sum_init = tensor.splat %f0 : tensor<1xf16>

        %sum_exp = scf.for %split = %c0 to %num_splits step %c1
            iter_args(%running_sum = %sum_init) -> tensor<1xf16> {

            %row = arith.addi %base_row, %split : index

            %logit_acc = ktdp.construct_access_tile %mid_o_view[%row, %c64] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 0 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x1xindex>

            %logit_tile = ktdp.load %logit_acc : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>
            %tlogic = tensor.extract %logit_tile[%c0_e, %c0_e] : tensor<1x1xf16>

            %diff = arith.subf %tlogic, %max_scalar : f16
            %diff_tile = tensor.splat %diff : tensor<1xf16>
            %exp_tile = math.exp %diff_tile : tensor<1xf16>

            %new_sum = arith.addf %running_sum, %exp_tile : tensor<1xf16>
            scf.yield %new_sum : tensor<1xf16>
        }

        // ═══ Pass 2: Weighted sum of V vectors ═══
        %acc_init = arith.constant dense<0.0> : tensor<1x64xf16>

        %acc = scf.for %split = %c0 to %num_splits step %c1
            iter_args(%running_acc = %acc_init) -> tensor<1x64xf16> {

            %row = arith.addi %base_row, %split : index

            // Load V
            %v_tile_acc = ktdp.construct_access_tile %mid_o_view[%row, %c0] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x64xindex>

            %v_tile = ktdp.load %v_tile_acc : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

            // Load logit and compute weight = exp(logit - max)
            %logit_acc = ktdp.construct_access_tile %mid_o_view[%row, %c64] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 0 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x1xindex>

            %logit_tile = ktdp.load %logit_acc : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>
            %tlogic = tensor.extract %logit_tile[%c0_e, %c0_e] : tensor<1x1xf16>

            %diff = arith.subf %tlogic, %max_scalar : f16
            %diff_tile = tensor.splat %diff : tensor<1xf16>
            %weight_tile = math.exp %diff_tile : tensor<1xf16>
            %weight_scalar = tensor.extract %weight_tile[%c0_e] : tensor<1xf16>
            %weight_2d = tensor.splat %weight_scalar : tensor<1x64xf16>

            // acc += weight * V
            %weighted_v = arith.mulf %weight_2d, %v_tile : tensor<1x64xf16>
            %new_acc = arith.addf %running_acc, %weighted_v : tensor<1x64xf16>

            scf.yield %new_acc : tensor<1x64xf16>
        }

        // output = acc / sum_exp
        %sum_scalar = tensor.extract %sum_exp[%c0_e] : tensor<1xf16>
        %sum_2d = tensor.splat %sum_scalar : tensor<1x64xf16>
        %final_out = arith.divf %acc, %sum_2d : tensor<1x64xf16>

        %out_acc = ktdp.construct_access_tile %output_view[%pair_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x64xf16> -> !ktdp.access_tile<1x64xindex>

        ktdp.store %final_out, %out_acc : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

        // lse = max + log(sum_exp)
        %log_sum = math.log %sum_exp : tensor<1xf16>
        %max_1d = tensor.splat %max_scalar : tensor<1xf16>
        %lse_val = arith.addf %max_1d, %log_sum : tensor<1xf16>

        %lse_acc = ktdp.construct_access_tile %lse_view[%pair_id] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<32xf16> -> !ktdp.access_tile<1xindex>

        ktdp.store %lse_val, %lse_acc : tensor<1xf16>, !ktdp.access_tile<1xindex>

        scf.yield
    }
    return
  }
}
