"""模块数值正确性校验（实验脚本开头调用）"""
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵不一致: {x} vs {xm}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(15.0, K / N)
    for _ in range(50):
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r)
        assert np.array_equal(u_hat[info_idx], payload)

    u4 = np.array([1, 0, 1, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u4)), 0.01)
    frozen4 = np.array([True, False, False, True], dtype=bool)
    u_sc = sc_decode(llr, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)

    if verbose:
        print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
