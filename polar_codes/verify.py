"""数值正确性校验"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, verify_sc_decoders
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    if not np.array_equal(x, [1, 1, 0, 1]):
        ok = False
        if verbose:
            print("编码器错误:", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        uh = sc_decode(llr, frozen_bits)
        if np.any(uh[info_idx] != u[info_idx]):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比测试失败: {sc_err}/100 帧错误")

    if not verify_sc_decoders(64, 30):
        ok = False
        if verbose:
            print("递归/非递归 SC 不一致")

    # L=1 SCL 等价 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
    usc = sc_decode_recursive(llr, frozen_bits)
    uscl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(usc, uscl):
        ok = False
        if verbose:
            print("SCL L=1 与 SC 不一致")

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    if not crc_check(bits, 8):
        ok = False
        if verbose:
            print("CRC 校验失败")

    if verbose:
        print("校验通过" if ok else "校验未全部通过")
    return ok


if __name__ == "__main__":
    run_all(True)
