"""单元测试：编码器、SC/SCL 译码正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    if verbose:
        print("✓ 编码器校验通过")

    # SC 递归 vs 非递归一致性
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.3) + rng.normal(0, 0.5, N)
        r1 = sc_decode_recursive(llr, frozen_bits)
        r2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(r1, r2), "递归与非递归 SC 不一致"
    if verbose:
        print("✓ SC 递归/非递归一致性校验通过")

    # 极低噪声下 SC 应无错
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"高 SNR SC 译码出现 {errors} 帧错误"
    if verbose:
        print("✓ 高 SNR SC 无损校验通过")

    # L=1 SCL 应等价于 SC
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
        sc_out = sc_decode(llr, frozen_bits)
        scl_out, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(sc_out, scl_out), "L=1 SCL 与 SC 不一致"
    if verbose:
        print("✓ L=1 SCL ≡ SC 路径度量校验通过")

    if verbose:
        print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
