"""单元测试与校验函数"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, construct_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有模块单元测试。"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = construct_generator_matrix(4)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print(f"[PASS] 编码器: u={u} -> x={x}")

    # GA 构造验证
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"[PASS] GA N=8: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[PASS] GA N=256 前20: {info256[:20]}")

    # SC 译码校验
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码错误: {sc_errors}/100"
    print(f"[PASS] SC 译码: 10dB 下 100 帧全部正确")

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, K / N)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 与 SC 不一致: {scl_errors}/50"
    print("[PASS] SCL L=1 等价于 SC")

    print("=" * 50)
    print("所有单元测试通过!")
    print("=" * 50)


if __name__ == '__main__':
    run_unit_tests()
