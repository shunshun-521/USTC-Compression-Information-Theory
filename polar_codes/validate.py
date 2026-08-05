"""极化码模块单元测试"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    # 编码器：矩阵法与蝶形法一致
    for N in [4, 8, 16]:
        u = np.random.default_rng(0).integers(0, 2, N)
        assert np.array_equal(
            polar_encode(u), polar_encode_matrix(u)
        ), f"编码器矩阵/蝶形不一致 N={N}"

    # 编码器已知向量（u=[0,0,1,1] 经 G_N 得 [0,0,1,1]）
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    # SC 译码无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        r = sc_decode(llr, frozen)
        if np.any(r[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码无损验证失败: {errors}/100 帧错误"

    # 递归与非递归 SC 一致
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
    r1 = sc_decode_recursive(llr, frozen)
    r2 = sc_decode(llr, frozen)
    assert np.array_equal(r1, r2), "递归与非递归 SC 不一致"

    # SCL L=1 等价于 SC
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.3)
    r_sc = sc_decode(llr, frozen)
    r_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(r_sc, r_scl), "SCL L=1 与 SC 不一致"

    # CRC 自洽
    info = rng.integers(0, 2, 20)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
