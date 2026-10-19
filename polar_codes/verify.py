"""单元测试：编码器与各译码器基础正确性"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from simulation import align_channel_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def _llr_from_u(u, sigma, rng):
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    return align_channel_llr(compute_llr(y, sigma), len(u))


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = _llr_from_u(u, sigma, rng)
        uh = sc_decode(llr, frozen)
        assert np.all(uh[info] == u[info]), "SC 高信噪比译码失败"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = _llr_from_u(u, sigma, rng)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应与 SC 一致"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    full = crc_encode(info, 8)
    assert crc_check(full, 8)
    full[-1] ^= 1
    assert not crc_check(full, 8)


def test_bp_noiseless_small():
    N = 8
    frozen = np.zeros(N, dtype=bool)
    bp = BPDecoder(N, frozen, max_iter=80)
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(12.0, 0.5)
    for _ in range(16):
        u = rng.integers(0, 2, N)
        llr = _llr_from_u(u, sigma, rng)
        uh, _ = bp.decode(llr)
        assert np.array_equal(uh, u), "BP 小码长高信噪比译码失败"


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless_small()
    print("verify.py: 全部测试通过")


if __name__ == "__main__":
    run_all()
