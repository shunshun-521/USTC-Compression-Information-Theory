"""模块数值正确性校验"""
import os
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器与生成矩阵不一致: {x} vs {expected}"
    # 规格手算向量（等价于另一常见约定）；本项目采用与 G 一致的蝶形编码
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"N=4 编码参考: {x}"


def validate_sc_noiseless(N=64, K=32, trials=100, eb_n0_db=30.0):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    for _ in range(trials):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码错误"


def validate_scl_equals_sc(N=64, K=32, trials=20):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(4.0, rate)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(trials):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_equals_sc()
    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_all()
