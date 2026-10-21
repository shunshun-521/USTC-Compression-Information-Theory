"""单元测试：编码器与译码器正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: 得到 {x}, 期望 {expected}")
    elif verbose:
        print("编码器校验通过")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 译码校验失败: {sc_err}/100 帧错误")
    elif verbose:
        print("SC 译码校验通过 (N=64, Eb/N0=10dB)")

    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    l1_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            l1_err += 1
    if l1_err > 0:
        ok = False
        if verbose:
            print(f"SCL L=1 与 SC 不一致: {l1_err}/50")
    elif verbose:
        print("SCL L=1 与 SC 等价校验通过")

    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_validation() else 1)
