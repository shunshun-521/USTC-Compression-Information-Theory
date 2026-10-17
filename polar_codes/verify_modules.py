"""各模块数值校验（实验脚本启动时调用）"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all_verifications():
    # 编码器：x = u @ G_N
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, np.asarray((u @ G) % 2).ravel()), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8:", info, frozen)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u_full[info_idx] = payload
        x = polar_encode(u_full)
        sigma = eb_n0_to_sigma(12.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat[info_idx], payload):
            ok += 1
    assert ok >= 99, f"SC 高信噪比测试失败: {ok}/100"

    # L=1 SCL 等价 SC
    u_full = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, K)
    u_full[info_idx] = payload
    x = polar_encode(u_full)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    # CRC
    info_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    coded = crc_encode(info_bits, 8)
    assert crc_check(coded, 8)

    print("verify_modules: 全部通过")


if __name__ == "__main__":
    run_all_verifications()
