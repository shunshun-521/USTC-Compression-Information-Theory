"""极化码模块数值正确性校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print(f"[PASS] 编码器校验: u={u} -> x={x}")


def validate_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"[INFO] N=8,K=4 info_indices={info}, frozen_indices={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[INFO] N=256,K=128 first20 info={info256[:20]}")


def validate_sc_decoder():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 失败 {errors}/100 帧"
    print("[PASS] SC 译码校验（N=64,K=32,100帧@10dB）")


def validate_scl_path_metric():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] 路径度量校验（SCL L=1 等价 SC）")


def validate_crc():
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("[PASS] CRC-8 校验")


def main():
    validate_encoder()
    validate_ga_construction()
    validate_sc_decoder()
    validate_scl_path_metric()
    validate_crc()
    print("\n所有校验通过。")


if __name__ == "__main__":
    main()
