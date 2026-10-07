"""模块数值校验"""
import numpy as np
from encoder import polar_encode, polar_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}, 期望 {(u @ G) % 2}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    fails = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(uh[info], u[info]):
            fails += 1
    assert fails == 0, f"SC 高信噪比测试失败: {fails}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_validations()
