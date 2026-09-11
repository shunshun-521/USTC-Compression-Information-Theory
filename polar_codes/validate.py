"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准蝶形编码（无最终比特倒序）：u=[1,0,1,1] -> x=[1,1,0,1]
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6]), f"GA N=8 错误: {info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("✓ GA 构造校验通过, N=256 info[:20] =", info256[:20])


def validate_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = 1e-9
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "SC 递归与非递归不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("✓ SC 译码校验通过 (100/100 帧正确)")


def validate_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = 1e-9
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 校验通过")


def validate_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.flip(coded), 8)
    print("✓ CRC 校验通过")


def main():
    validate_encoder()
    validate_construction()
    validate_crc()
    validate_sc_lossless()
    validate_scl_equiv_sc()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()
