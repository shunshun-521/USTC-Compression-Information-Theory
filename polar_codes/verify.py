"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def verify_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def verify_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 译码在极低噪声下失败"
    print("SC 译码校验通过（100 帧无误）")


def verify_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(456)
    for _ in range(50):
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"
    print("SCL 路径度量校验通过（L=1 等价 SC）")


def verify_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8,K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256,K=128: info 前20={info256[:20]}")


def run_all():
    verify_encoder()
    verify_sc_decoder()
    verify_scl_path_metric()
    verify_construction()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()
