"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    # 编码器：u @ G_N（与蝶形+比特倒序编码一致）
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    assert np.array_equal(x2, [1, 0, 1, 1]), f"编码器错误 (u2): {x2}"

    # SC 无损（N=64, K=32, 高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        errors += np.sum(u_hat[info_idx] != u[info_idx])
    assert errors == 0, f"SC 高信噪比校验失败，信息位错误数={errors}"

    # L=1 SCL 等价 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("所有单元校验通过。")


if __name__ == "__main__":
    run_validations()
