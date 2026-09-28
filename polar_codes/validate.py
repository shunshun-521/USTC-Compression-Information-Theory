"""模块数值校验"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    assert err == 0, f"SC 校验失败: {err}/100 帧错误"

    scl = SCLDecoder(N, frozen, list_size=1)
    err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh, _ = scl.decode(llr)
        if not np.array_equal(uh, u):
            err += 1
    assert err == 0, f"SCL L=1 应等价 SC: {err} 错误"

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
