"""模块数值正确性校验（在各 run_exp*.py 开头调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_paths_equivalent
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validations():
    from encoder import polar_generator_matrix

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"

    llr_test = np.array([1.0, -2.0, 0.5, -0.5, 2.0, -1.0, 0.0, 3.0])
    fb8 = np.zeros(8, dtype=bool)
    assert sc_paths_equivalent(llr_test, fb8), "递归与非递归 SC 不一致"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, K / N)
    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    print("所有校验通过。")
