"""极化码仿真单元测试"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    """运行所有模块正确性校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_tx = np.zeros(N, dtype=int)
        u_tx[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u_tx)) + rng.normal(0, sigma, N)
        u_hat = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(u_hat[info_idx], u_tx[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码校验失败: {errors}/100 帧有错"

    mismatches = 0
    for _ in range(20):
        u_tx = np.zeros(N, dtype=int)
        u_tx[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u_tx)) + rng.normal(0, sigma, N), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不等价: {mismatches}/20"

    payload = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(payload, 8), "CRC 校验失败"

    print("所有单元测试通过。")
