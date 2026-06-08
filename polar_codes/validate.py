"""单元测试：验证各模块正确性"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    # 编码器校验（与 G 矩阵一致）
    N = 4
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F)
    br = bit_reversal_permutation(N)
    G_br = G[br, :]
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.mod(u @ G_br, 2)
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"

    # SC 译码校验（Eb/N0=10dB，N=64,K=32，100 帧无错误）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 译码失败"

    # 递归与非递归 SC 一致
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    assert np.array_equal(
        sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits)
    )

    # L=1 的 SCL 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, sc_decode(llr, frozen_bits))

    # CRC 校验
    info = rng.integers(0, 2, size=20)
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)
    bad = payload.copy()
    bad[-1] ^= 1
    assert not crc_check(bad, 8)

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
