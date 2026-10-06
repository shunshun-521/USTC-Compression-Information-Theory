"""模块数值校验（实验脚本可 import）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equivalent_sc


def validate_encoder():
    """编码器：自洽性 + 与 u@G_N 一致（含比特倒序）。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]])
    G = np.kron(F, F)
    B = np.zeros((4, 4), dtype=int)
    rev = bit_reversal_permutation(4)
    for i in range(4):
        B[i, rev[i]] = 1
    GN = (B @ G) % 2
    x_ref = (u @ GN) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"


def validate_sc_noiseless(num_trials=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rev = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)
    sigma = 1e-6
    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        llr_dec = llr[rev]
        u_hat = sc_decode(llr_dec, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码错误"


def validate_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rev = bit_reversal_permutation(N)
    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = rng.normal(0, 2.0, size=N)
        llr_dec = llr[rev]
        u1 = sc_decode(llr_dec, frozen_bits)
        u2 = sc_decode_recursive(llr_dec, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"


def validate_scl_path_metric():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rev = bit_reversal_permutation(N)
    rng = np.random.default_rng(3)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)[rev]
        assert scl_equivalent_sc(llr, frozen_bits), "L=1 SCL 应等价 SC"


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_sc_recursive_match()
    validate_scl_path_metric()
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()
