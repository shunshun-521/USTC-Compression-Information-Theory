"""极化码模块单元测试"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])

    # SC 高 SNR
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    # 无损信道 LLR 下 SC 应完全正确
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * (1.0 - 2.0 * x)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errs += 1
    assert errs == 0, f"SC 无损测试失败: {errs} 帧错误"

    # SCL L=1 等价 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    llr = 50.0 * (1.0 - 2.0 * x)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
