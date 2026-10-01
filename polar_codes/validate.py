"""模块数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准 B_N F^{⊗n} 编码结果
    # 标准 B_N F^{⊗n} 编码（含比特倒序）
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def validate_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh = sc_decode(llr, frozen)
        uh_r = sc_decode_recursive(llr, frozen)
        assert np.array_equal(uh, u), "非递归 SC 译码错误"
        assert np.array_equal(uh_r, u), "递归 SC 译码错误"


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"


def main():
    validate_encoder()
    validate_sc()
    validate_scl_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    main()
