"""模块数值正确性校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def _encoder_test():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 Kronecker 生成矩阵 G=F^{\otimes 2} 一致（无输出倒序）
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def _sc_lossless_test():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u, u_hat)


def _scl_l1_test():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 0.1)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def run_all():
    _encoder_test()
    _sc_lossless_test()
    _scl_l1_test()
    print("verify.py: 所有校验通过")


if __name__ == "__main__":
    run_all()
