"""
模块正确性验证脚本
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[PASS] 编码器校验")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), f"SC 译码失败 seed={seed}"
    print("[PASS] SC 译码校验（极低噪声, 100 帧）")


def validate_scl_equivalence():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for seed in range(20):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), f"SCL(L=1) 与 SC 不一致 seed={seed}"
    print("[PASS] SCL(L=1) 等价于 SC")


def validate_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    print("[PASS] CRC 校验")


def validate_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits)

    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.random.default_rng(0).integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无损译码失败"
    print("[PASS] BP 无损译码校验")


def validate_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info={info256[:20]}")
    print("[PASS] GA 构造输出")


def main():
    validate_encoder()
    validate_ga_construction()
    validate_sc_noiseless()
    validate_scl_equivalence()
    validate_crc()
    validate_bp_noiseless()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()
