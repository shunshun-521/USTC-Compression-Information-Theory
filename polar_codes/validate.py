"""极化码模块数值校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(5.0, 0.5))
    u_scl, _ = scl.decode(llr)
    u_sc = sc_decode(llr, frozen)
    assert np.array_equal(u_scl, u_sc), "SCL(L=1) 与 SC 不一致"

    info = np.array([1, 0, 1, 1, 0, 1, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(coded[:-1], 8)

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
