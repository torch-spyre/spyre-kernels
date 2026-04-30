// SwiGLU (SiLU-gated linear unit) forward kernel in KTDP dialect
//
// Algorithm:
//   gate = X[row, 0:d]
//   up   = X[row, d:2*d]
//   gate_f32 = float32(gate)
//   up_f32   = float32(up)
//   sigmoid_gate = 1.0 / (1.0 + exp(-gate_f32))
//   silu_gate = sigmoid_gate * gate_f32
//   gate_clamped = min(silu_gate, limit)       // = -max(-silu_gate, -limit)
//   up_clamped   = clamp(up_f32, -limit, limit) // = -max(-max(up, -limit), -limit)
//   result = float16(gate_clamped * up_clamped)
//
// Note: The interpreter lacks arith.minimumf, so min is implemented as
// -max(-a, -b). This is equivalent for finite values.
//
// Concrete sizes: 32 rows, d=1024, input [32, 2048], output [32, 1024]
// Grid: [32, 1] — one core per row

module {
  func.func @silu_and_mul_kernel(
      %X: index,       // input  32x2048xf16 (gate || up concatenated)
      %Y: index,       // output 32x1024xf16
      %d: index        // half-width: 1024
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %f0_f32 = arith.constant 0.0 : f32
    %f1_f32 = arith.constant 1.0 : f32
    %neg1_f32 = arith.constant -1.0 : f32
    %limit_f32 = arith.constant 7.0 : f32
    %neg_limit_f32 = arith.constant -7.0 : f32

    // Input view: [32, 2048]
    %X_view = ktdp.construct_memory_view %X, sizes: [32, 2048], strides: [2048, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 2047 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x2048xf16>

    // Output view: [32, 1024]
    %Y_view = ktdp.construct_memory_view %Y, sizes: [32, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x1024xf16>

    scf.for %row = %core_id to %c32 step %c32 : index {

        // Load gate: X[row, 0:1024]
        %gate_acc = ktdp.construct_access_tile %X_view[%row, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x2048xf16> -> !ktdp.access_tile<1x1024xindex>

        %gate_f16 = ktdp.load %gate_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

        // Load up: X[row, 1024:2048]
        %up_acc = ktdp.construct_access_tile %X_view[%row, %d] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x2048xf16> -> !ktdp.access_tile<1x1024xindex>

        %up_f16 = ktdp.load %up_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

        // Upcast to f32
        %gate = arith.extf %gate_f16 : tensor<1x1024xf16> to tensor<1x1024xf32>
        %up = arith.extf %up_f16 : tensor<1x1024xf16> to tensor<1x1024xf32>

        // sigmoid(gate) = 1.0 / (1.0 + exp(-gate))  (f32)
        %f0_block = tensor.splat %f0_f32 : tensor<1x1024xf32>
        %neg_gate = arith.subf %f0_block, %gate : tensor<1x1024xf32>
        %exp_neg = math.exp %neg_gate : tensor<1x1024xf32>
        %f1_block = tensor.splat %f1_f32 : tensor<1x1024xf32>
        %one_plus_exp = arith.addf %f1_block, %exp_neg : tensor<1x1024xf32>
        %sigmoid = arith.divf %f1_block, %one_plus_exp : tensor<1x1024xf32>

        // silu(gate) = sigmoid(gate) * gate  (f32)
        %silu = arith.mulf %sigmoid, %gate : tensor<1x1024xf32>

        // gate_clamped = min(silu, limit) = -max(-silu, -limit)
        %neg1_block = tensor.splat %neg1_f32 : tensor<1x1024xf32>
        %neg_silu = arith.mulf %silu, %neg1_block : tensor<1x1024xf32>
        %neg_limit_block = tensor.splat %neg_limit_f32 : tensor<1x1024xf32>
        %neg_gate_max = arith.maximumf %neg_silu, %neg_limit_block : tensor<1x1024xf32>
        %gate_clamped = arith.mulf %neg_gate_max, %neg1_block : tensor<1x1024xf32>

        // up_clamped = clamp(up, -limit, limit) = min(max(up, -limit), limit)
        //   max(up, -limit) first:
        %up_lower = arith.maximumf %up, %neg_limit_block : tensor<1x1024xf32>
        //   min(up_lower, limit) = -max(-up_lower, -limit)
        %neg_up_lower = arith.mulf %up_lower, %neg1_block : tensor<1x1024xf32>
        %neg_up_max = arith.maximumf %neg_up_lower, %neg_limit_block : tensor<1x1024xf32>
        %up_clamped = arith.mulf %neg_up_max, %neg1_block : tensor<1x1024xf32>

        // result = gate_clamped * up_clamped  (f32)
        %result_f32 = arith.mulf %gate_clamped, %up_clamped : tensor<1x1024xf32>

        // Downcast to f16
        %result = arith.truncf %result_f32 : tensor<1x1024xf32> to tensor<1x1024xf16>

        // Store output
        %Y_acc = ktdp.construct_access_tile %Y_view[%row, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<32x1024xf16> -> !ktdp.access_tile<1x1024xindex>

        ktdp.store %result, %Y_acc : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

        scf.yield
    }
    return
  }
}
