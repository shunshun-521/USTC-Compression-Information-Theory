"""极化码仿真公共验证与辅助函数"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行编码器与译码器单元测试"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < 4:
        G = np.kron(G, F)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x_code = polar_encode(u_full)
        llr = compute_llr(bpsk_modulate(x_code), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_full[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下失败 {errors}/100 帧"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_full)), 0.001)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_scl, u_sc), "L=1 的 SCL 应与 SC 等价"

    print("所有单元测试通过。")
