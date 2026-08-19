"""单元测试"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """单元测试验证各模块正确性"""
    print("=" * 50)
    print("运行单元测试...")
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"
    print("  编码器测试通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码测试失败: {errors}/100 错误"
    print("  SC 译码测试通过 (N=64, Eb/N0=10dB, 100帧)")

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_test)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
    print("  SCL(L=1) 路径度量测试通过")
    print("单元测试全部通过。")
    print("=" * 50)
