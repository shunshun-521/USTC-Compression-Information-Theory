"""模块数值校验"""
import os
import numpy as np

from construction import ga_construction, ga_construction as _
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器校验（与生成矩阵一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"

    # SC 无损验证
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        if np.array_equal(sc_decode(llr, frozen), u):
            ok += 1
    assert ok == 100, f"SC 译码失败: {ok}/100"

    # L=1 SCL 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
