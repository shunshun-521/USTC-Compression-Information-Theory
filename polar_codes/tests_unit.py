"""极化码模块单元测试（各实验脚本启动时调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equals_sc
from encoder import polar_encode


def run_unit_tests():
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器：u=[1,0,1,1] -> x=[1,0,1,1]（G = B_N F^{⊗n}）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    # SC 无损（N=4, K=2, 高 SNR）
    N, K = 4, 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧错误"

    # SCL L=1 等价 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = [1, 1]
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    assert scl_equals_sc(N, frozen_bits, llr), "SCL(L=1) 与 SC 不等价"

    print("单元测试全部通过。")


if __name__ == "__main__":
    run_unit_tests()
