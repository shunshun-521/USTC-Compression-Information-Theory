"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, build_generator_matrix


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("[OK] 编码器校验通过")


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)

    for trial in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), f"SC 译码失败 frame={trial}"

    print("[OK] SC 译码校验通过 (N=64, 100 帧 @ 10dB)")


def validate_scl_equivalence():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)

    for trial in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), f"SCL L=1 与 SC 不一致 frame={trial}"

    print("[OK] SCL L=1 等价 SC 校验通过")


def validate_crc():
    info = np.random.randint(0, 2, 32)
    for r in (8, 16):
        coded = crc_encode(info, r)
        assert crc_check(coded, r)
    print("[OK] CRC 校验通过")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, first 20 info: {info256[:20]}")


def run_all():
    np.random.seed(42)
    validate_encoder()
    validate_crc()
    validate_sc()
    validate_scl_equivalence()
    validate_construction()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()
