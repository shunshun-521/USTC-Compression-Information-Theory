"""极化码仿真实验公共校验与工具。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行编码与译码单元测试。"""
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f'编码器错误: {x}'

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    brp = bit_reversal_permutation(N)
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u_full)), sigma, rng)
        llr = compute_llr(y, sigma)[brp].astype(np.float32)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], payload):
            sc_errors += 1
    assert sc_errors == 0, f'SC 译码在 10dB 下错误帧数: {sc_errors}'

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u_full)), 0.01, rng)
        llr = compute_llr(y, 0.01)[brp].astype(np.float32)
        uh_scl, _ = scl.decode(llr)
        uh_sc = sc_decode(llr, frozen)
        assert np.array_equal(uh_scl, uh_sc), 'L=1 SCL 应与 SC 等价'
    print('单元测试全部通过。')
