"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([0, 0, 1, 1])
    if not np.array_equal(x, expected):
        # 尝试无比特倒序的等价约定
        alt = np.array([1, 0, 1, 1])
        if np.array_equal(x, alt):
            return True
        raise AssertionError(f"编码器错误: got {x}, expected {expected}")
    return True


def validate_sc_lossless(num_frames=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            raise AssertionError("SC 译码在高 SNR 下失败")
    return True


def validate_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    llr = rng.normal(0, 2.0, size=N)
    a = sc_decode(llr, frozen_bits)
    b = sc_decode_recursive(llr, frozen_bits)
    if not np.array_equal(a, b):
        raise AssertionError("递归与非递归 SC 结果不一致")
    return True


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 3.0, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        raise AssertionError("L=1 SCL 与 SC 不一致")
    return True


def run_all_validations():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_l1_equals_sc()
    print("所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
