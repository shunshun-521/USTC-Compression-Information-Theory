"""模块数值校验（各实验脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码测试向量不符: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 高信噪比译码失败"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
