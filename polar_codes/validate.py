"""模块数值校验（仿真脚本可 import）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print("编码器与生成矩阵不一致:", x, x_ref)
    u2 = np.array([0, 0, 1, 1])
    if not np.array_equal(polar_encode(u2), [0, 0, 1, 1]):
        ok = False
        if verbose:
            print("编码器 u=[0,0,1,1] 校验失败")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    if errors > 0:
        ok = False
        if verbose:
            print(f"SC 无损测试失败: {errors}/100 帧错误")

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    payload = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        ok = False
        if verbose:
            print("SCL L=1 与 SC 不一致")

    u_rec = sc_decode_recursive(llr, frozen_bits)
    if not np.array_equal(u_sc, u_rec):
        ok = False
        if verbose:
            print("递归与非递归 SC 不一致")

    if verbose:
        print("validate:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_unit_tests() else 1)
