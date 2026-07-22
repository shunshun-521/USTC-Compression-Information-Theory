"""共享单元测试与验证函数。"""
import os
import numpy as np
from encoder import polar_encode, polar_encode_matrix
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有模块单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = 0

    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_sent[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下出现 {errors} 个错误帧"

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, 0.5, N), 0.5)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen)
        assert np.array_equal(u_scl, u_sc), "L=1 的 SCL 应与 SC 等价"

    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")


def quick_env():
    return os.environ.get("POLAR_QUICK", "0") == "1"
