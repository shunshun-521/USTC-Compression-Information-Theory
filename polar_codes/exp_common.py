"""极化码仿真实验公共配置与校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix

POLAR_QUICK = os.environ.get('POLAR_QUICK', '0') == '1'


def run_unit_tests():
    """仿真前数值正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f'编码器错误: {x}'

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        x_full = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x_full), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), 'SC 无损译码失败'

    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), 'L=1 SCL 与 SC 不一致'

    print('单元测试通过。')


def quick_params(default_n_list, default_max_frames, default_min_errors, default_eb_range):
    """根据 POLAR_QUICK 环境变量缩减仿真规模。"""
    if not POLAR_QUICK:
        return default_n_list, default_max_frames, default_min_errors, default_eb_range
    n_list = default_n_list[:1] if len(default_n_list) > 1 else default_n_list
    eb_range = default_eb_range[:2] if len(default_eb_range) > 2 else default_eb_range
    return n_list, 500, 10, eb_range
