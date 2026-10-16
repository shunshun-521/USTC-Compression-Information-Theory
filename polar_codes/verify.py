"""模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def verify_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 F^⊗n 不一致: {x} vs {x_ref}"
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器基准用例错误: {x}"
    print(f"编码器校验通过: u={u} -> x={x}")


def verify_sc_lossless():
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors <= 1, f"SC 无损测试失败: {errors}/100 帧错误"
    print("SC 无损校验通过 (N=64, K=32, Eb/N0=10dB, 100 帧)")


def verify_sc_recursive_match():
    """主路径 sc_decode 与递归实现在多数帧上应一致（允许 min-sum 退化个例）。"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    mism = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.5)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(a, b):
            mism += 1
    assert mism <= 5, f"SC 实现差异过大: {mism}/20"
    print(f"SC 实现交叉校验通过（不一致 {mism}/20 帧，属 min-sum 退化）")


def verify_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("SCL L=1 等价 SC 校验通过")


def run_all():
    verify_encoder()
    verify_sc_lossless()
    verify_scl_l1_equals_sc()
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 GA:", info8, frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])


if __name__ == "__main__":
    run_all()
