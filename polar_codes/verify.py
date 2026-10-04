"""模块数值校验（仿真脚本启动前调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_all_verifications():
    # 编码器：与 u @ G_N（蝶形 + 比特倒序）一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # GA 构造 smoke
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    # CRC round-trip
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
    with_crc = crc_encode(info, 8)
    assert crc_check(with_crc, 8)

    # SC 无损（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(20.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # L=1 SCL 等价 SC
    N = 64
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("所有校验通过。")


if __name__ == "__main__":
    run_all_verifications()
