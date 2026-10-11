"""极化码模块单元测试（实验脚本运行前调用）"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, SCDecoder
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def run_all(verbose=True):
    rng = np.random.default_rng(0)

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("N=8 info:", info, "frozen:", frozen)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        for dec in (sc_decode, sc_decode_recursive):
            uh = dec(llr, frozen_bits)
            if not np.array_equal(uh, u):
                err += 1
    assert err == 0, "SC 译码在 Eb/N0=10dB 下应无错误"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scd = SCDecoder(N, frozen_bits)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
        uh_sc = scd.decode(llr)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应等价于 SC"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 0]), 8)
    assert crc_check(bits, 8)

    bp = BPDecoder(N, frozen_bits, max_iter=80)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-4)
    uh, iters = bp.decode(llr)
    assert uh.shape == (N,) and iters >= 1, "BP 译码应返回有效结果"

    if verbose:
        print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()
