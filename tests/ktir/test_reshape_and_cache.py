"""Validate KV Cache Reshape KTDP MLIR against NumPy reference."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM  # 512
BLOCK_SIZE = 16
NUM_BLOCKS = 4
CACHE_SLOTS = NUM_BLOCKS * BLOCK_SIZE  # 64


def reshape_and_cache_ref(
    key: np.ndarray,
    value: np.ndarray,
    slot_mapping: np.ndarray,
) -> tuple:
    """NumPy reference: copy key/value rows to cache slots.

    key, value: [32, 512]
    slot_mapping: [32] with int values in [0, 64)
    Returns: key_cache[64, 512], value_cache[64, 512]
    """
    key_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)
    value_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)

    for t in range(NUM_TOKENS):
        slot = int(slot_mapping[t])
        key_cache[slot] = key[t]
        value_cache[slot] = value[t]

    return key_cache, value_cache


def test_reshape_and_cache_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    key = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    value = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    slot_mapping = rng.choice(CACHE_SLOTS, size=NUM_TOKENS, replace=False).astype(np.float16)

    key_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)
    value_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "reshape_and_cache_kernel",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        block_size=BLOCK_SIZE,
    )
    result_kc = outputs["key_cache"]
    result_vc = outputs["value_cache"]

    expected_kc, expected_vc = reshape_and_cache_ref(key, value, slot_mapping)

    np.testing.assert_allclose(
        result_kc.astype(np.float32), expected_kc.astype(np.float32),
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        result_vc.astype(np.float32), expected_vc.astype(np.float32),
        rtol=0, atol=0,
    )
    max_err_k = np.max(np.abs(result_kc.astype(np.float32) - expected_kc.astype(np.float32)))
    max_err_v = np.max(np.abs(result_vc.astype(np.float32) - expected_vc.astype(np.float32)))
    print(f"PASS: key max err = {max_err_k:.6f}, value max err = {max_err_v:.6f}")


def test_reshape_sequential_slots():
    """Sequential slot_mapping (identity permutation)."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    key = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    value = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    slot_mapping = np.arange(NUM_TOKENS, dtype=np.float16)

    key_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)
    value_cache = np.zeros((CACHE_SLOTS, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "reshape_and_cache_kernel",
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        block_size=BLOCK_SIZE,
    )

    np.testing.assert_array_equal(outputs["key_cache"][:NUM_TOKENS], key)
    np.testing.assert_array_equal(outputs["value_cache"][:NUM_TOKENS], value)
    print("PASS: sequential slot mapping")


if __name__ == "__main__":
    test_reshape_and_cache_ktir()
    test_reshape_sequential_slots()
    print("\nAll KV Cache Reshape KTIR validation tests passed!")
