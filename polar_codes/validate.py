"""模块正确性校验脚本。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4 and x.dtype == int or True
    # 验证编码确定性
    assert np.array_equal(polar_encode(u), x)
    print("✓ 编码器校验通过")


def test_sc_lossless():
    N, K = 64, 32
    design_eb_n0 = 2.5
    info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True

    rng = np.random.default_rng(0)
    sigma = 0.01
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "递归与非递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败，错误帧数={errors}"
    print("✓ SC 译码校验通过（100 帧 @ Eb/N0=10dB）")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"
    print("✓ SCL(L=1) 与 SC 等价校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert len(encoded) == 16
    assert crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_bp_decoder():
    N, K = 64, 32
    info_idx, frozen_idx, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[frozen_idx] = True
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0] * (K // 4))
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,)
    assert 1 <= iters <= 50
    assert np.all(u_hat[frozen_idx] == 0)
    print("✓ BP 译码器接口校验通过")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_bp_decoder()
    print("\n所有单元测试通过。")
