"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8,K=4 info:", info, "frozen:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256,K=128 info (first 20):", info256[:20])

    N, K = 64, 64
    frozen = np.zeros(N, bool)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 1.0)
    ok = 0
    for _ in range(100):
        u = rng.integers(0, 2, N)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        if np.array_equal(sc_decode(compute_llr(y, sigma), frozen), u):
            ok += 1
    assert ok >= 80, f"SC 高信噪比测试失败: {ok}/100"

    N = 32
    frozen = np.zeros(N, bool)
    scl = SCLDecoder(N, frozen, list_size=1)
    u = rng.integers(0, 2, N)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(12.0, 1.0))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
