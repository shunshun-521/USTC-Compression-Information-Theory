"""模块数值正确性校验（仿真前必须通过）"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all(verbose=True):
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"

    # GA 构造 sanity
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("N=8 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=256 info first 20:", info256[:20])

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "SC 递归与非递归不一致"
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"
    if verbose:
        print("SC 无损测试通过 (N=64, 100 frames)")

    # SCL L=1 等价 SC
    u4 = np.array([1, 0, 1, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u4)), 0.01)
    frozen4 = np.array([True, False, False, True])
    u_sc = sc_decode(llr, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    # CRC
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)

    if verbose:
        print("verify.py: 全部校验通过")
    return True


if __name__ == "__main__":
    run_all()
