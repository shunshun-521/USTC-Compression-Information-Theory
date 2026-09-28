"""模块级单元测试（各实验脚本运行前调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 高 SNR 校验失败，错误帧数={sc_errors}"

    u0 = np.zeros(N, dtype=int)
    u0[info_idx] = rng.integers(0, 2, K)
    x0 = polar_encode(u0)
    llr0 = compute_llr(bpsk_modulate(x0), 0.01)
    u_sc = sc_decode(llr0, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr0)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"

    if verbose:
        print("单元测试全部通过。")


if __name__ == "__main__":
    run_unit_tests()
