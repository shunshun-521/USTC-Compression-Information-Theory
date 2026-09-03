"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print(f"  u={u} -> x={x}")
    print("✓ 编码器校验通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 first 20 info: {info256[:20]}")
    print("✓ 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    design_eb = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = 100.0 * bpsk_modulate(x)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
        u_hat_r = sc_decode_recursive(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
    assert errors == 0, f"SC 无损验证失败: {errors}/100 错误"
    print("✓ SC 译码校验通过（100 帧无错误，无噪声）")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = 100.0 * bpsk_modulate(x)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块单元测试")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_crc()
    print("=" * 50)
    print("全部测试通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all()
