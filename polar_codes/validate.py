"""模块数值正确性校验"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"
    print("编码器校验通过 (GF(2) G_N)")


def validate_sc_lossless():
    N, K = 8, 4
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    assert _noiseless_indices_valid(N, info_idx, num_checks=40), "构造的信息位未通过自检"
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    br = bit_reversal_permutation(N)

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(80):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)[br]
        u_hat = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_rec), "非递归与递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"
    print("SC 译码校验通过 (100 帧 @10dB)")


def validate_scl_equals_sc():
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        br = bit_reversal_permutation(N)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )[br]
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 路径度量校验通过")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("CRC 校验通过")


def validate_ga_reference():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert _noiseless_indices_valid(8, info)
    print("GA 构造参考值通过", "info=", info)


def _noiseless_indices_valid(N, info_indices, num_checks=8):
    from construction import _noiseless_indices_valid as check

    return check(N, info_indices, num_checks=num_checks)


def run_all():
    validate_encoder()
    validate_ga_reference()
    validate_crc()
    validate_sc_lossless()
    validate_scl_equals_sc()
    print("全部校验通过。")


if __name__ == "__main__":
    run_all()
