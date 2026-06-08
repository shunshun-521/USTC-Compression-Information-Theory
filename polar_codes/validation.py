"""单元测试：验证各模块数值正确性。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行编码器、SC/SCL 校验，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    if verbose:
        print("编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {sc_errors} 帧错误"
    if verbose:
        print("SC 译码校验通过 (100 帧无误)")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_hat, _ = scl.decode(llr)
        assert np.array_equal(
            u_hat, sc_decode(llr, frozen_bits)
        ), "L=1 的 SCL 应与 SC 等价"
    if verbose:
        print("SCL 路径度量校验通过 (L=1 等价于 SC)")
