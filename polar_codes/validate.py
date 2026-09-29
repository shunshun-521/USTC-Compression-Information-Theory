"""模块数值校验（各实验脚本开头调用）"""
import numpy as np

from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    if not np.array_equal(x, x_ref):
        raise AssertionError(f"编码器与生成矩阵不一致: {x} vs {x_ref}")


def _channel_llr_for_decode(y, sigma, N):
    """信道 LLR（与码字比特顺序一致）"""
    return compute_llr(y, sigma)


def validate_sc_lossless(num_trials=100, N=64, K=32):
    from construction import ga_construction

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = s + rng.normal(0, sigma, size=N)
        llr = _channel_llr_for_decode(y, sigma, N)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            u_hat_r = sc_decode_recursive(llr, frozen_bits)
            if np.array_equal(u_hat_r[info_idx], u[info_idx]):
                raise AssertionError("非递归 SC 与递归结果不一致")
            raise AssertionError("SC 译码失败（高 SNR）")


def validate_scl_equals_sc(N=64):
    from construction import ga_construction

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, size=N)
        llr = _channel_llr_for_decode(y, sigma, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("L=1 SCL 与 SC 不一致")


def run_all_validations():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_equals_sc()
    print("所有校验通过。")
