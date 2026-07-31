"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有模块校验"""
    # 编码器校验：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x} vs {u @ G % 2}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"GA N=8: info={info8}, frozen={frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"GA N=256: first 20 info indices = {info256[:20]}")

    # SC 无损校验
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 校验失败: {sc_errors}/100 帧有错"

    # SCL L=1 等价 SC
    rng = np.random.default_rng(456)
    scl_mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_mismatch += 1
    assert scl_mismatch == 0, f"SCL L=1 与 SC 不一致: {scl_mismatch}/50"

    # CRC 校验
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8), "CRC-8 校验失败"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
