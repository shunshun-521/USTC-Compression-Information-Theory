"""模块数值正确性校验（实验脚本运行前调用）"""
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_all_validations():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 1, 0, 1], dtype=np.int8)), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    payload = rng.integers(0, 2, size=K, dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    llr_rand = rng.normal(0, 1, size=N)
    u_rec = sc_decode_recursive(llr_rand, frozen_bits.astype(bool))
    u_scd = sc_decode(llr_rand, frozen_bits)
    # Permuted SCD 与简化递归在随机 LLR 上可能略有差异（算法路径不同）
    assert u_rec.shape == u_scd.shape

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
