"""模块数值正确性校验（各实验脚本可 import）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(123)
    fails = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            fails += 1
    assert fails == 0, f"SC 高信噪比测试失败: {fails}/100"

    frozen = frozen_bits.astype(bool)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
    uh_sc, _ = sc_decode(llr, frozen_bits), None
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
