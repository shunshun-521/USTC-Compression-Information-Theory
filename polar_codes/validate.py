"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: butterfly {x}, matrix {x_mat}"
    # 与 G_N=B_N F^{\\otimes n} 行向量乘法一致（部分教材手算示例记号不同）
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_ga_n8():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0


def test_sc_noiseless():
    from encoder import bit_reversal_permutation

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, N // 2)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)[br]
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh[info_idx], payload)


def test_sc_recursive_match():
    N = 128
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, N)
    a = sc_decode(llr, frozen)
    b = sc_decode_recursive(llr, frozen)
    assert np.array_equal(a, b), "非递归 SC 与递归 SC 不一致"


def test_scl_l1_equals_sc():
    from encoder import bit_reversal_permutation

    N = 64
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3, N)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"


def test_bp_single():
    N = 16
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    uh, _ = BPDecoder(N, frozen, max_iter=20).decode(llr[br])
    assert np.array_equal(uh[info_idx], u[info_idx])


if __name__ == "__main__":
    test_encoder()
    test_ga_n8()
    test_sc_recursive_match()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_bp_single()
    print("All validate.py tests passed.")
