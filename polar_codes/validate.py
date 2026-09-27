"""模块数值正确性校验（各实验脚本开头调用）。"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=np.int32)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下失败帧数: {errors}"

    payload = rng.integers(0, 2, size=K)
    u_sent = np.zeros(N, dtype=np.int32)
    u_sent[info_idx] = payload
    x = polar_encode(u_sent)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_rec = sc_decode_recursive(llr, frozen_bits)
    u_nr = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 不一致"

    u_sent = np.zeros(N, dtype=np.int32)
    u_sent[info_idx] = payload
    x = polar_encode(u_sent)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    if verbose:
        print("所有单元测试通过。")
