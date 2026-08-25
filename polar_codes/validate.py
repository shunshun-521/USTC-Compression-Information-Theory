"""
单元测试与数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder


def test_encoder():
    """编码器：验证与生成矩阵一致"""
    def gf2_mat_pow(F, n):
        G = np.array([[1]], dtype=int)
        for _ in range(n):
            G = np.kron(G, F)
        return G

    from encoder import bit_reversal_permutation

    for n in [2, 3, 4]:
        N = 2 ** n
        F = np.array([[1, 0], [1, 1]], dtype=int)
        F_n = gf2_mat_pow(F, n)
        br = bit_reversal_permutation(N)
        B = np.zeros((N, N), dtype=int)
        for i in range(N):
            B[br[i], i] = 1
        G = B @ F_n

        rng = np.random.default_rng(0)
        for _ in range(50):
            u = rng.integers(0, 2, N)
            x_enc = polar_encode(u)
            x_mat = u @ G % 2
            assert np.array_equal(x_enc, x_mat), f"编码器与 G 矩阵不一致: {x_enc} vs {x_mat}"

    # 规范示例：u=[0,0,1,1] -> x=[0,0,1,1]
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"


def test_sc_lossless():
    """极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"


def test_scl_equals_sc():
    """单路径 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(4.0, K / N)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不一致"


def test_bp_zero_noise():
    """BP 译码零噪声测试"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen)

    rng = np.random.default_rng(3)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        uh, _ = bp.decode(llr)
        assert np.array_equal(uh[info_idx], u[info_idx]), "BP 零噪声译码失败"


def test_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128
    print("N=8 info:", info8, "frozen:", frozen8)
    print("N=256 first 20 info:", info256[:20])


def main():
    print("运行单元测试...")
    test_encoder()
    print("  编码器: OK")
    test_sc_lossless()
    print("  SC 无损: OK")
    test_scl_equals_sc()
    print("  SCL L=1 == SC: OK")
    test_bp_zero_noise()
    print("  BP 零噪声: OK")
    test_construction()
    print("  GA 构造: OK")
    print("全部测试通过。")


if __name__ == '__main__':
    main()
