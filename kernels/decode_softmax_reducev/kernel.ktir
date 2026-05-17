// Decode softmax + reduceV (stage 2) kernel in KTDP dialect
//
// Online softmax merge — single-pass approach matching the block-pointer Triton:
//   for each split:
//     n_e_max = max(tlogic, e_max)
//     old_scale = exp(e_max - n_e_max)
//     acc = acc * old_scale + exp(tlogic - n_e_max) * V
//     e_sum = e_sum * old_scale + exp(tlogic - n_e_max)
//     e_max = n_e_max
//   output = acc / e_sum
//   lse = e_max + log(e_sum)
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

        // Initial values for online softmax
        %emax_init = tensor.splat %neg_inf : tensor<1xf16>
        %esum_init = tensor.splat %f0 : tensor<1xf16>
        %acc_init = arith.constant dense<0.0> : tensor<1x64xf16>

        // Single-pass online softmax merge
        %emax_final, %esum_final, %acc_final = scf.for %split = %c0 to %num_splits step %c1
            iter_args(%e_max = %emax_init, %e_sum = %esum_init, %acc = %acc_init)
            -> (tensor<1xf16>, tensor<1xf16>, tensor<1x64xf16>) {

            %row = arith.addi %base_row, %split : index

            // Load V vector for this split
            %v_tile_acc = ktdp.construct_access_tile %mid_o_view[%row, %c0] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x64xindex>

            %tv = ktdp.load %v_tile_acc : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

            // Load logit scalar for this split
            %logit_acc = ktdp.construct_access_tile %mid_o_view[%row, %c64] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 0 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<128x65xf16> -> !ktdp.access_tile<1x1xindex>

            %logit_tile = ktdp.load %logit_acc : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>
            %tlogic = tensor.extract %logit_tile[%c0_e, %c0_e] : tensor<1x1xf16>
            %tlogic_1d = tensor.splat %tlogic : tensor<1xf16>

            // n_e_max = max(tlogic, e_max)
            %n_e_max = arith.maximumf %tlogic_1d, %e_max : tensor<1xf16>

            // old_scale = exp(e_max - n_e_max)
            %diff_old = arith.subf %e_max, %n_e_max : tensor<1xf16>
            %old_scale = math.exp %diff_old : tensor<1xf16>

            // exp_logic = exp(tlogic - n_e_max)
            %diff_new = arith.subf %tlogic_1d, %n_e_max : tensor<1xf16>
            %exp_logic = math.exp %diff_new : tensor<1xf16>

            // acc = acc * old_scale + exp_logic * tv
            %old_scale_scalar = tensor.extract %old_scale[%c0_e] : tensor<1xf16>
            %old_scale_2d = tensor.splat %old_scale_scalar : tensor<1x64xf16>
            %acc_scaled = arith.mulf %acc, %old_scale_2d : tensor<1x64xf16>

            %exp_logic_scalar = tensor.extract %exp_logic[%c0_e] : tensor<1xf16>
            %exp_logic_2d = tensor.splat %exp_logic_scalar : tensor<1x64xf16>
            %weighted_v = arith.mulf %exp_logic_2d, %tv : tensor<1x64xf16>

            %new_acc = arith.addf %acc_scaled, %weighted_v : tensor<1x64xf16>

            // e_sum = e_sum * old_scale + exp_logic
            %esum_scaled = arith.mulf %e_sum, %old_scale : tensor<1xf16>
            %new_esum = arith.addf %esum_scaled, %exp_logic : tensor<1xf16>

            scf.yield %n_e_max, %new_esum, %new_acc : tensor<1xf16>, tensor<1xf16>, tensor<1x64xf16>
        }

        // output = acc / e_sum
        %esum_scalar = tensor.extract %esum_final[%c0_e] : tensor<1xf16>
        %esum_2d = tensor.splat %esum_scalar : tensor<1x64xf16>
        %final_out = arith.divf %acc_final, %esum_2d : tensor<1x64xf16>

        %out_acc = ktdp.construct_access_tile %output_view[%pair_id, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x64xf16> -> !ktdp.access_tile<1x64xindex>

        ktdp.store %final_out, %out_acc : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

        // lse = e_max + log(e_sum)
        %log_esum = math.log %esum_final : tensor<1xf16>
        %lse_val = arith.addf %emax_final, %log_esum : tensor<1xf16>

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
