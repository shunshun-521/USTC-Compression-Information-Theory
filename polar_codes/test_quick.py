"""快速 SC 译码测试脚本"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder

# 编码器校验：N=4, u=[1,0,1,1] -> x=[1,1,0,1]
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
print("编码器测试通过")

# SC 译码校验（无损验证）
N, K = 64, 32
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

rng = np.random.default_rng(0)
sigma = eb_n0_to_sigma(10.0, K / N)
errors = 0
mismatch = 0
for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits.astype(bool))
    u_hat_r = sc_decode_recursive(llr, frozen_bits.astype(bool))
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
        errors += 1
    if not np.array_equal(u_hat_r, u_hat):
        mismatch += 1

print(f"SC 高信噪比测试: {100 - errors}/100 正确, 递归一致: {mismatch == 0}")

# 路径度量校验：L=1 SCL 应等价于 SC
u = np.zeros(N, dtype=int)
u[info_idx] = rng.integers(0, 2, K)
x = polar_encode(u)
y = awgn_channel(bpsk_modulate(x), sigma, rng)
llr = compute_llr(y, sigma)
u_sc = sc_decode(llr, frozen_bits.astype(bool))
u_scl, _ = SCLDecoder(N, frozen_bits.astype(bool), list_size=1).decode(llr)
print("L=1 SCL == SC:", np.array_equal(u_sc, u_scl))

# 无噪声测试
errors = 0
for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    u_hat = sc_decode(llr, frozen_bits.astype(bool))
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
        errors += 1
print(f"SC 无噪声测试: {100 - errors}/100 正确")
