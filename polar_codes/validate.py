"""极化码模块数值正确性校验"""
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
from simulation import run_simulation


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1 - 2 * x) * 100.0
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)
    print("SC 无损校验通过 (N=64, 100 帧)")


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1 - 2 * x) * 50.0
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("L=1 SCL 等价 SC 校验通过")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("CRC 校验通过")


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8,K=4 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256,K=128 info[:20]={info256[:20]}")


def validate_awgn_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr):
        return sc_decode(llr, frozen_bits), None

    results = run_simulation(
        N, K, [15.0], decoder, 'sc',
        max_frames=500, min_errors=500, seed=0,
        info_indices=info_idx, frozen_bits=frozen_bits,
        verbose=False,
    )
    assert results[0]['num_errors'] == 0, "高信噪比 SC 译码应无错误"
    print("AWGN SC 高信噪比校验通过")


if __name__ == '__main__':
    validate_encoder()
    validate_construction()
    validate_crc()
    validate_sc_noiseless()
    validate_scl_l1_equals_sc()
    validate_awgn_sc()
    print("\n全部校验通过。")
