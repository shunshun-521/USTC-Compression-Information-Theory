"""极化码模块单元测试。"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from simulation import prepare_decoder_llr


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"
    if verbose:
        print("✓ 编码器校验通过")

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), f"N=8 GA 错误: {info8}"
    if verbose:
        print("✓ GA 构造 N=8,K=4 校验通过")

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print(f"✓ N=256 前 20 个信息位索引: {info256[:20]}")

    # SC 无损校验
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码错误"
    if verbose:
        print("✓ SC 译码 100 帧无损校验通过 (Eb/N0=10dB)")

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    if verbose:
        print("✓ SCL L=1 等价 SC 校验通过")

    # CRC 校验
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 编码/检验失败"
    if verbose:
        print("✓ CRC-8 校验通过")

    if verbose:
        print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
