"""极化码模块单元测试（各实验脚本开头调用）"""
import os
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    # 1. 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])  # 与生成矩阵 G_N 一致
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"
    print("  [PASS] 编码器校验")

    # 2. GA 构造校验
    info256, _, _ = ga_construction(256, 128, 2.5)
    ref = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
    assert np.array_equal(info256[:20], ref), f"GA 构造错误: {info256[:20]}"
    print("  [PASS] GA 构造校验")

    # 3. CRC 校验
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 编码/校验失败"
    print("  [PASS] CRC 校验")

    # 4. SC 译码无损验证（高 SNR）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = 0.001  # 近似无噪信道
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "非递归与递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧错误"
    print("  [PASS] SC 译码无损验证 (N=64, 100帧)")

    # 5. SCL L=1 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("  [PASS] SCL L=1 等价 SC")

    print("所有单元测试通过!\n")


if __name__ == "__main__":
    run_unit_tests()
