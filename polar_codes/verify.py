"""
实验脚本公共单元测试
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """编码器、SC 译码、SCL 路径度量校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在 Eb/N0=10dB 下失败 {sc_errors}/100 帧"

    rng = np.random.default_rng(456)
    scl_mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-8)
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1, crc_length=0).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_mismatch += 1
    assert scl_mismatch == 0, f"L=1 SCL 与 SC 不一致 {scl_mismatch}/50 帧"

    print("单元测试全部通过。")


if __name__ == "__main__":
    run_unit_tests()
