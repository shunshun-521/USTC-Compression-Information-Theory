"""模块数值正确性校验"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, sc_decode_via_scl
from encoder import polar_encode


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    u4 = np.zeros(4, dtype=int)
    u4[1], u4[3] = 1, 1
    x = polar_encode(u4)
    if not np.array_equal(x, [0, 0, 1, 1]):
        ok = False
        if verbose:
            print("编码器错误:", x)

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    if err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {err}/100")

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    uh_sc, _ = scl.decode(llr)
    if not np.array_equal(uh_sc, sc_decode(llr, frozen)):
        ok = False
        if verbose:
            print("L=1 SCL 与 SC 不一致")

    if verbose and ok:
        print("verify.py: 所有单元测试通过")
    return ok


if __name__ == "__main__":
    run_unit_tests()
