"""模块数值校验（各实验脚本开头调用）。"""
import os

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_verifications(quick=False):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [3, 5, 6, 7]), info8
    assert np.array_equal(frozen8, [0, 1, 2, 4]), frozen8

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    frames = 10 if quick else 50
    ok = 0
    for _ in range(frames):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(u_hat[info_idx], payload))
    assert ok >= max(1, int(0.85 * frames)), f"SC 高信噪比通过率过低: {ok}/{frames}"

    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(
        compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 0.1)
    )
    assert u_scl is not None

    print("所有校验通过。")


if __name__ == "__main__":
    run_all_verifications(quick=os.environ.get("POLAR_QUICK") == "1")
