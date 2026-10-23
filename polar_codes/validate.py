"""模块正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validate():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N), rng)
        llr = compute_llr(y, eb_n0_to_sigma(10.0, K / N))
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errs += 1
    assert errs == 0, f"SC 无损校验失败: {errs}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.05)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 等价"

    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_validate()
