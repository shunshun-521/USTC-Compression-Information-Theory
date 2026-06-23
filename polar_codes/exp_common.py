"""实验脚本公共工具"""
import os

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def run_unit_validation():
    """各实验脚本开头的单元测试"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x_full = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x_full), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u_full[info_idx])

    for _ in range(20):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x_full = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x_full), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def quick_mode_enabled():
    return os.environ.get("POLAR_QUICK", "0") == "1"


def sim_params(default_max_frames, default_min_errors, default_eb_range):
    if quick_mode_enabled():
        return {
            "max_frames": min(default_max_frames, 2000),
            "min_errors": min(default_min_errors, 20),
            "eb_n0_range": default_eb_range[::2]
            if len(default_eb_range) > 6
            else default_eb_range,
        }
    return {
        "max_frames": default_max_frames,
        "min_errors": default_min_errors,
        "eb_n0_range": default_eb_range,
    }
