"""Validate KV Cache Reshape KTDP MLIR against vLLM kernel."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.vllm.reshape_and_cache.wrapper import reshape_and_cache

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "vllm" / "reshape_and_cache" / "kernel.ktir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM  # 512
BLOCK_SIZE = 16
NUM_BLOCKS = 4
CACHE_SLOTS = NUM_BLOCKS * BLOCK_SIZE  # 64


def vllm_reference(
    key_flat: np.ndarray,
    value_flat: np.ndarray,
    slot_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run vLLM reshape_and_cache kernel on GPU.

    KTIR layout: key/value [T, H*D], caches [slots, H*D], slot_mapping [T] i64
    Wrapper layout: key/value [T, H, D], caches [num_blocks, block_size, H, D], slot_mapping [T] int64
    Returns: key_cache, value_cache as [slots, H*D] (KTIR layout)
    """
    key = torch.from_numpy(
        key_flat.reshape(NUM_TOKENS, NUM_HEADS, HEAD_DIM).astype(np.float16)
    ).cuda()
    value = torch.from_numpy(
        value_flat.reshape(NUM_TOKENS, NUM_HEADS, HEAD_DIM).astype(np.float16)
    ).cuda()
    key_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16
    )
    value_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16
    )
    slot_mapping_t = torch.from_numpy(slot_mapping.astype(np.int64)).cuda()

    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping_t)

    kc = key_cache.reshape(CACHE_SLOTS, FLAT_DIM).cpu().numpy()
    vc = value_cache.reshape(CACHE_SLOTS, FLAT_DIM).cpu().numpy()
    return kc, vc


def test_reshape_and_cache_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    key = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    value = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    slot_mapping = rng.choice(CACHE_SLOTS, size=NUM_TOKENS, replace=False).astype(np.int64)

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

    expected_kc, expected_vc = vllm_reference(key, value, slot_mapping)

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
    slot_mapping = np.arange(NUM_TOKENS, dtype=np.int64)

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


