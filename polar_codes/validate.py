"""极化码模块单元测试与数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行所有单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [3, 5, 6, 7]), f"N=8 info_indices 错误: {info8}"
    assert np.array_equal(frozen8, [0, 1, 2, 4]), f"N=8 frozen_indices 错误: {frozen8}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
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
    assert sc_errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {sc_errors}/100 帧错误"

    # SCL L=1 应等价于 SC
    scl_errors = 0
    rng = np.random.default_rng(456)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL(L=1) 与 SC 不一致: {scl_errors}/50 帧"

    # BP 低噪声校验
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    bp_errors = 0
    rng = np.random.default_rng(789)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat, u):
            bp_errors += 1
    assert bp_errors == 0, f"BP 低噪声译码失败: {bp_errors}/20 帧"

    if verbose:
        print("所有单元测试通过。")
    return True
