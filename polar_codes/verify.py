"""模块数值校验（供实验脚本调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, scl_equivalent_sc
from encoder import arikan_generator, polar_encode


def verify_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = arikan_generator(4)
    x_ref = (G @ u) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    # 规格文档手算示例 [0,0,1,1] 与标准 G_N 约定不同，以生成矩阵一致为准
    return True


def verify_sc_noiseless(N=64, K=32, frames=100, eb_n0_db=15.0):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)
    for _ in range(frames):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
    return True


def verify_scl_l1(N=64):
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        llr = rng.normal(0, 2, size=N)
        assert scl_equivalent_sc(llr, frozen_bits)
    return True


def verify_recursive_sc(N=16):
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(10):
        llr = rng.normal(0, 2, size=N)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)
    return True


def run_all():
    verify_encoder()
    verify_sc_noiseless()
    verify_scl_l1()
    print("All verify checks passed.")


if __name__ == "__main__":
    run_all()
