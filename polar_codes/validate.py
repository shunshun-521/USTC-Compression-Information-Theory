"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    if not np.array_equal(x, expected):
        ok = False
        print(f"编码器错误: got {x}, expected {expected}")
    elif verbose:
        print("编码器校验通过")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print(f"N=8 GA info_indices: {info8}")
        print(f"N=8 GA frozen_indices: {frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print(f"N=256 GA first 20 info_indices: {info256[:20]}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(u, uh):
            sc_errors += 1
    if sc_errors > 0:
        ok = False
        print(f"SC 译码校验失败: {sc_errors}/100 帧错误")
    elif verbose:
        print("SC 译码校验通过 (Eb/N0=10dB, 100帧)")

    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            scl_errors += 1
    if scl_errors > 0:
        ok = False
        print(f"SCL(L=1) 与 SC 不一致: {scl_errors}/50")
    elif verbose:
        print("SCL L=1 等价 SC 校验通过")

    payload = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    coded = crc_encode(payload, 8)
    if not crc_check(coded, 8):
        ok = False
        print("CRC 校验失败")
    elif verbose:
        print("CRC 校验通过")

    return ok


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
