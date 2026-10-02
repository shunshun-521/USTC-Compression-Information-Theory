"""模块数值校验（编码、SC、SCL L=1）"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    if not np.array_equal(x, [1, 1, 0, 1]):
        ok = False
        if verbose:
            print(f"编码器错误: 期望 [1,1,0,1], 得到 {x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info
        s = bpsk_modulate(polar_encode(u_sent))
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    if errors > 0:
        ok = False
        if verbose:
            print(f"SC 高 SNR 校验失败: {errors}/100 帧错误")

    rng = np.random.default_rng(1)
    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info
        s = bpsk_modulate(polar_encode(u_sent))
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_sc, u_rec):
            ok = False
            if verbose:
                print("递归/非递归 SC 不一致")
            break

    rng = np.random.default_rng(2)
    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info
        s = bpsk_modulate(polar_encode(u_sent))
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            ok = False
            if verbose:
                print("SCL L=1 与 SC 不一致")
            break

    if verbose:
        print("verify.py:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_unit_tests() else 1)
