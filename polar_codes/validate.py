"""模块数值校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validate():
    # 编码器：与生成矩阵一致
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"

    # GA 构造打印
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20])

    # SC 高信噪比
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(15.0, K / N)
    bit_err = 0
    frames = 50 if not os.environ.get("POLAR_QUICK") else 10
    for _ in range(frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        bit_err += np.sum(uh[info_idx] != u[info_idx])
    ber = bit_err / (frames * K)
    assert ber < 0.45, f"SC 高信噪比 BER 过高: {ber:.4f}"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不一致"

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validate()
