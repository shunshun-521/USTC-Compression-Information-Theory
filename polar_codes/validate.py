"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode, scl_path_metric_check
from encoder import polar_encode
from simulation import run_simulation


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f'编码器错误: {x}, 期望 {expected}'
    print('[PASS] encoder')


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print('N=8 info:', info, 'frozen:', frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256 first 20 info:', info256[:20])
    assert len(info) == 4 and len(frozen) == 4
    print('[PASS] ga_construction')


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f'SC 无损译码失败: {errors}/100'
    print('[PASS] sc noiseless')


def test_sc_high_snr():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(15.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors == 0, f'高 SNR SC 译码失败: {errors}/100'
    print('[PASS] sc high snr')


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        ok, _ = scl_path_metric_check(N, frozen, llr)
        assert ok, 'L=1 SCL 与 SC 不等价'
    print('[PASS] scl l=1')


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)
    payload_corrupt = payload.copy()
    payload_corrupt[-1] ^= 1
    assert not crc_check(payload_corrupt, 8)
    print('[PASS] crc')


def test_bp_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    bp = BPDecoder(N, frozen, max_iter=50)
    rng = np.random.default_rng(3)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f'BP 无损失败: {errors}/20'
    print('[PASS] bp noiseless')


def main():
    print('=' * 60)
    print('极化码模块校验')
    print('=' * 60)
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print('=' * 60)
    print('全部校验通过')
    print('=' * 60)


if __name__ == '__main__':
    main()
