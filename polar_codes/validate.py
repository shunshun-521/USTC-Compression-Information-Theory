"""模块正确性快速校验（仿真脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"


def validate_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 译码失败"
        u_ref = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u_hat, u_ref), "递归/非递归 SC 不一致"


def validate_scl_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def run_all():
    validate_encoder()
    validate_sc()
    validate_scl_equals_sc()
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("GA N=8:", info8, frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 info (first 20):", info256[:20])
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()
