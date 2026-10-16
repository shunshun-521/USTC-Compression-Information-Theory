"""模块数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, verify_sc_implementations
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    verify_sc_implementations(64, 30)

    # SC 无损
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh, u), "SC 高信噪比译码失败"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(8.0, 0.5))
        u_sc = sc_decode_recursive(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
