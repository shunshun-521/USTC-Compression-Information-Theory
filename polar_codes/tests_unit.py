"""极化码模块单元测试。"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_full[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧有错误"

    mismatches = 0
    for _ in range(50):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不等价: {mismatches}/50"

    print("所有单元测试通过。")
