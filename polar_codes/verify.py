"""模块数值正确性校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel, reorder_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    # GA 构造打印
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info_indices:", info8)
    print("N=8, K=4, frozen_indices:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices[:20]:", info256[:20])

    # SC 高信噪比
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = reorder_llr_for_decoder(compute_llr(bpsk_modulate(polar_encode(u)), 1e-9))
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh, u):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100 帧错误"

    # SCL L=1 等价 SC
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, K)
    llr = reorder_llr_for_decoder(
        compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(4.0, 0.5))
    )
    uh_sc = sc_decode(llr, frozen_bits)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "SCL(L=1) 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
