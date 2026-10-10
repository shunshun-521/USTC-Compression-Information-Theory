"""模块数值校验（实验脚本运行前调用）"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100 帧错误"

    u4 = np.array([1, 0, 1, 1])
    llr4 = compute_llr(bpsk_modulate(polar_encode(u4)), sigma)
    uh_sc = sc_decode(llr4, np.zeros(4, dtype=int))
    uh_scl, _ = SCLDecoder(4, np.zeros(4, dtype=int), list_size=1).decode(llr4)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_tests()
