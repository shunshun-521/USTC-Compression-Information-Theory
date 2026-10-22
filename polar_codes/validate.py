"""模块正确性校验（仿真前调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validate(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err_sc = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        if not np.array_equal(sc_decode(llr, fb), u):
            err_sc += 1
    assert err_sc == 0, f"SC 高信噪比测试失败: {err_sc}/100 帧错误"

    scl = SCLDecoder(N, fb, list_size=1, crc_length=0)
    err_scl = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat, _ = scl.decode(llr)
        if not np.array_equal(u_hat, u):
            err_scl += 1
    assert err_scl == 0, f"L=1 SCL 应与 SC 一致: {err_scl}/20 帧错误"

    if verbose:
        print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validate()
