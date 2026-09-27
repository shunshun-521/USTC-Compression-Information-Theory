"""模块数值校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_validate(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    if not np.array_equal(x, (u @ G) % 2):
        ok = False
        if verbose:
            print("编码器校验失败:", x)

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=8 info:", info8, "frozen:", frozen8)
        print("N=256 info first20:", info256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            sc_err += 1
    if sc_err > 0:
        ok = False
    if verbose:
        print(f"SC 高 SNR 帧错误数: {sc_err}/100")

    mism = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mism += 1
    if mism > 0:
        ok = False
    if verbose:
        print(f"SCL L=1 与 SC 不一致帧数: {mism}/50")

    payload = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    if not crc_check(payload, 8):
        ok = False
        if verbose:
            print("CRC 自校验失败")

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    success = run_validate(True)
    sys.exit(0 if success else 1)
