#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run all 9 tritokti kernels through both regex and mlir-fe parsers.

Usage:
    uv run python scripts/test_tritokti.py /path/to/tritokti/kernels
"""
import sys
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings("ignore")

from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

KERNELS = {
    "rms_norm": {
        "func": "rms_norm_fwd",
        "kwargs": lambda: dict(
            X=np.random.default_rng(42).standard_normal((32, 4096)).astype(np.float16),
            W=np.ones(4096, dtype=np.float16),
            Y=np.zeros((32, 4096), dtype=np.float16),
            N=4096,
            eps=np.float16(1e-5),
            BLOCK_SIZE=1024,
        ),
    },
    "silu_and_mul": {
        "func": "silu_and_mul_kernel",
        "kwargs": lambda: dict(
            X=np.random.default_rng(42).standard_normal((32, 2048)).astype(np.float16),
            Y=np.zeros((32, 1024), dtype=np.float16),
            d=1024,
        ),
    },
    "ranks": {
        "func": "ranks_kernel",
        "kwargs": lambda mlir: _ranks_kwargs(mlir),
    },
    "log_softmax": {
        "func": "log_softmax_kernel",
        "kwargs": lambda mlir: _log_softmax_kwargs(mlir),
    },
    "decode_softmax_reducev": {
        "func": "decode_softmax_reducev_kernel",
        "kwargs": lambda: _decode_softmax_kwargs(),
    },
    "merge_attn_states": {
        "func": "merge_attn_states_kernel",
        "kwargs": lambda: dict(
            prefix_output=np.random.default_rng(42).standard_normal((32, 512)).astype(np.float16),
            suffix_output=np.random.default_rng(43).standard_normal((32, 512)).astype(np.float16),
            prefix_lse=np.random.default_rng(44).standard_normal((8, 32)).astype(np.float16),
            suffix_lse=np.random.default_rng(45).standard_normal((8, 32)).astype(np.float16),
            output=np.zeros((32, 512), dtype=np.float16),
            num_heads=8,
        ),
    },
    "mrope": {
        "func": "mrope_kernel",
        "kwargs": lambda: dict(
            q=np.random.default_rng(42).standard_normal((32, 512)).astype(np.float16),
            k=np.random.default_rng(43).standard_normal((32, 512)).astype(np.float16),
            cos_ptr=np.random.default_rng(44).standard_normal((32, 32)).astype(np.float16),
            sin_ptr=np.random.default_rng(45).standard_normal((32, 32)).astype(np.float16),
            num_q_heads=8,
            num_kv_heads=8,
        ),
    },
    "reshape_and_cache": {
        "func": "reshape_and_cache_kernel",
        "kwargs": lambda: dict(
            key=np.random.default_rng(42).standard_normal((32, 512)).astype(np.float16),
            value=np.random.default_rng(43).standard_normal((32, 512)).astype(np.float16),
            key_cache=np.zeros((64, 512), dtype=np.float16),
            value_cache=np.zeros((64, 512), dtype=np.float16),
            slot_mapping=np.arange(32, dtype=np.int64),
            block_size=16,
        ),
    },
    "prefill_attention": {
        "func": "prefill_attention_kernel",
        "kwargs": lambda: _prefill_kwargs(),
    },
}


def _ranks_kwargs(mlir):
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((32, 4096)).astype(np.float16)
    token_ids = rng.integers(0, 4096, size=32).astype(np.int64)
    if "token_ids" in mlir:
        return dict(logits=logits, token_ids=token_ids, output=np.zeros(32, dtype=np.float16), vocab_size=4096, BLOCK_SIZE=1024)
    ref_logits = np.array([logits[i, token_ids[i]] for i in range(32)], dtype=np.float16)
    return dict(logits=logits, ref_logits=ref_logits, output=np.zeros(32, dtype=np.float16), vocab_size=4096, BLOCK_SIZE=1024)


def _log_softmax_kwargs(mlir):
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((32, 4096)).astype(np.float16)
    topk_ids = rng.integers(0, 4096, size=(32, 8)).astype(np.int64)
    if "topk_ids" in mlir:
        return dict(logits=logits, topk_ids=topk_ids, output=np.zeros((32, 8), dtype=np.float16), vocab_size=4096, topk=8, BLOCK_SIZE=1024)
    topk_logits = np.array([[logits[i, topk_ids[i, k]] for k in range(8)] for i in range(32)], dtype=np.float16)
    return dict(logits=logits, topk_logits=topk_logits, output=np.zeros((32, 8), dtype=np.float16), vocab_size=4096, topk=8, BLOCK_SIZE=1024)


def _decode_softmax_kwargs():
    rng = np.random.default_rng(42)
    mid_o = np.zeros((128, 65), dtype=np.float16)
    mid_o[:, :64] = (rng.standard_normal((128, 64)) * 0.1).astype(np.float16)
    mid_o[:, 64] = (rng.uniform(0.1, 1.0, 128)).astype(np.float16)
    return dict(
        mid_o=mid_o,
        lse_out=np.zeros(32, dtype=np.float16),
        output=np.zeros((32, 64), dtype=np.float16),
        num_splits=4,
    )


def _prefill_kwargs():
    rng = np.random.default_rng(42)
    seq_len, num_heads, head_dim = 16, 4, 64
    Q = rng.standard_normal((seq_len, num_heads * head_dim)).astype(np.float16)
    K = rng.standard_normal((seq_len, num_heads * head_dim)).astype(np.float16)
    V = rng.standard_normal((seq_len, num_heads * head_dim)).astype(np.float16)
    mask = np.zeros((seq_len, seq_len), dtype=np.float16)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = np.float16(-1e4)
    return dict(
        q_ptr=Q, k_ptr=K, v_ptr=V,
        output_ptr=np.zeros_like(Q),
        causal_mask_ptr=mask,
        num_heads=num_heads,
    )


def run_kernel(name, mlir_path, func_name, kwargs, parser_name, parser):
    try:
        interp = KTIRInterpreter(parser=parser) if parser else KTIRInterpreter()
        interp.load(open(mlir_path).read())
        outputs = interp.execute_function(func_name, **kwargs)
        return "PASS", outputs
    except Exception as e:
        return f"FAIL — {e}", None


def compare_outputs(regex_out, mlirfe_out, rtol=1e-2, atol=1e-2):
    if regex_out is None or mlirfe_out is None:
        return None
    r_vals = [v for v in regex_out.values() if isinstance(v, np.ndarray)]
    m_vals = [v for v in mlirfe_out.values() if isinstance(v, np.ndarray)]
    r_keys = [k for k, v in regex_out.items() if isinstance(v, np.ndarray)]
    if len(r_vals) != len(m_vals):
        return [f"output count mismatch: {len(r_vals)} vs {len(m_vals)}"]
    mismatches = []
    for i, (a, b) in enumerate(zip(r_vals, m_vals)):
        name = r_keys[i]
        if a.shape != b.shape:
            mismatches.append(f"{name}: shape {a.shape} vs {b.shape}")
            continue
        maxdiff = np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
        if not np.allclose(a.astype(np.float32), b.astype(np.float32), rtol=rtol, atol=atol):
            mismatches.append(f"{name}: max diff {maxdiff:.6f}")
    return mismatches


def _get_kwargs(cfg, mlir):
    kwargs_fn = cfg["kwargs"]
    import inspect
    if len(inspect.signature(kwargs_fn).parameters) > 0:
        return kwargs_fn(mlir)
    return kwargs_fn()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/tritokti/kernels")
        sys.exit(1)

    kernels_dir = Path(sys.argv[1])

    print(f"{'Kernel':<28} {'regex':<8} {'mlir-fe':<8} {'values':<8}")
    print("-" * 52)

    for name, cfg in KERNELS.items():
        mlir_path = kernels_dir / name / "kernel.ktir.mlir"
        if not mlir_path.exists():
            print(f"{name:<28} {'SKIP':<8} {'SKIP':<8} {'—':<8}")
            continue

        mlir = open(mlir_path).read()

        kwargs = _get_kwargs(cfg, mlir)
        regex_result, regex_out = run_kernel(name, mlir_path, cfg["func"], kwargs, "regex", None)

        kwargs = _get_kwargs(cfg, mlir)
        mlirfe_result, mlirfe_out = run_kernel(name, mlir_path, cfg["func"], kwargs, "mlir-fe", MLIRFrontendParser())

        mismatches = compare_outputs(regex_out, mlirfe_out)

        r = "✅" if regex_result == "PASS" else "❌"
        m = "✅" if mlirfe_result == "PASS" else "❌"
        if mismatches is None:
            v = "—"
        elif len(mismatches) == 0:
            v = "✅"
        else:
            v = "❌"
        print(f"{name:<28} {r:<8} {m:<8} {v:<8}")
        if regex_result != "PASS":
            print(f"  regex:   {regex_result}")
        if mlirfe_result != "PASS":
            print(f"  mlir-fe: {mlirfe_result}")
        if mismatches:
            for msg in mismatches:
                print(f"  values:  {msg}")

        # Print first output values for manual comparison
        out = regex_out or mlirfe_out
        if out:
            for k, v in out.items():
                if isinstance(v, np.ndarray) and v.size > 0:
                    flat = v.flatten()
                    print(f"  {k}[:4] = {flat[:min(4, len(flat))]}")

    print()
    print("values = regex vs mlir-fe output comparison (— when either parser fails)")


if __name__ == "__main__":
    main()
