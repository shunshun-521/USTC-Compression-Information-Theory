"""单元测试与数值校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validations():
    # 编码器
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 1, 0, 1]), f"编码器错误: {x}"

    # SC 小规模冒烟测试（N=8）
    N, K = 8, 4
    info = np.array([0, 1, 3, 7])
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(4.0, K / N)
    err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode_recursive(llr, frozen)
        if np.any(uh[info] != u[info]):
            err += 1
    assert err < 40, f"SC 冒烟测试失败: {err}/50"

    # SCL L=1 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), sigma)
    u_sc = sc_decode_recursive(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


if __name__ == "__main__":
    run_validations()
    print("validations passed")
