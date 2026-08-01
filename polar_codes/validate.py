#!/usr/bin/env python3
"""单元测试：验证极化码各模块正确性"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check

print('=== 编码器校验 ===')
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 0, 1, 1]), f'编码器错误: {x}'
print('编码器校验通过')

print('\n=== GA 构造校验 ===')
info, frozen, _ = ga_construction(8, 4, 2.5)
print(f'N=8: info={info}, frozen={frozen}')
info256, _, _ = ga_construction(256, 128, 2.5)
print(f'N=256 前20个 info_indices: {info256[:20]}')

print('\n=== SC 译码校验（递归 vs 非递归）===')
N, K = 64, 32
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

rng = np.random.default_rng(0)
for _ in range(20):
    u = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, K)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), '递归与非递归 SC 不一致'

print('SC 递归/非递归一致性校验通过')

print('\n=== SC 低噪声校验 ===')
errors = 0
for seed in range(100):
    rng = np.random.default_rng(seed)
    u = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, K)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], payload):
        errors += 1
assert errors == 0, f'SC 低噪声校验失败: {errors} 帧错误'
print('SC 低噪声校验通过 (100/100)')

print('\n=== SCL L=1 等价 SC ===')
for seed in range(50):
    rng = np.random.default_rng(seed)
    u = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, K)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.05)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), f'SCL L=1 与 SC 不一致 seed={seed}'

print('SCL L=1 等价 SC 校验通过')

print('\n=== CRC 校验 ===')
info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
coded = crc_encode(info_bits, 8)
assert crc_check(coded, 8), 'CRC 校验失败'
print('CRC 校验通过')

print('\n所有单元测试通过!')
