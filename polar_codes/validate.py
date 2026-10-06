"""模块数值校验（实验脚本开头调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print(f"编码器错误: x={x}, G@u={x_ref}")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_ok = 0
    for _ in range(100):
        msg = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=np.int8)
        u[info] = msg
        s = bpsk_modulate(polar_encode(u))
        llr = compute_llr(awgn_channel(s, sigma, rng), sigma)
        uh = sc_decode(llr, frozen)
        if np.array_equal(uh[info], msg):
            sc_ok += 1
    if sc_ok < 100:
        ok = False
        if verbose:
            print(f"SC 高信噪比校验失败: {sc_ok}/100 帧正确")

    scl = SCLDecoder(N, frozen, list_size=1)
    u_test = np.zeros(N, dtype=np.int8)
    u_test[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_test)), eb_n0_to_sigma(8.0, K / N))
    uh_sc, _ = sc_decode(llr, frozen), None
    uh_scl, _ = scl.decode(llr)
    if not np.array_equal(uh_sc, uh_scl):
        ok = False
        if verbose:
            print("L=1 SCL 与 SC 不一致")

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    run_validation()
