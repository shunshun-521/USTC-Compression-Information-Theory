"""模块数值校验（实验脚本开头调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造打印
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh = sc_decode(llr, frozen)
        uh_r = sc_decode_recursive(llr, frozen)
        assert np.array_equal(uh, u)
        assert np.array_equal(uh_r, u)

    # SCL L=1 等价 SC
    dec = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = dec.decode(llr)
    assert np.array_equal(uh_sc, uh_scl)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
