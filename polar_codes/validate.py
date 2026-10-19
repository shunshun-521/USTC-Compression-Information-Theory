"""模块数值正确性校验（各实验脚本导入运行）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print(f"编码器错误: {x} vs {x_ref}")
    else:
        if verbose:
            print(f"编码器校验通过: u={u} -> x={x}")

    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(payload, u_hat[info_idx]):
            errors += 1
    if errors > 0:
        ok = False
        if verbose:
            print(f"SC 无损校验失败: {errors}/100 帧错误")
    else:
        if verbose:
            print("SC 无损校验通过 (N=64, Eb/N0=10dB, 100 帧)")

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    if not np.array_equal(u_hat, u_hat_r):
        ok = False
        if verbose:
            print("递归与非递归 SC 不一致")
    else:
        if verbose:
            print("递归/非递归 SC 一致")

    N2, K2 = 32, 16
    info2, _, _ = ga_construction(N2, K2, design)
    fb2 = np.ones(N2, dtype=int)
    fb2[info2] = 0
    payload2 = rng.integers(0, 2, size=K2)
    u2 = np.zeros(N2, dtype=int)
    u2[info2] = payload2
    x2 = polar_encode(u2)
    llr2 = compute_llr(bpsk_modulate(x2), 0.01)
    u_sc = sc_decode(llr2, fb2)
    u_scl, _ = SCLDecoder(N2, fb2, list_size=1).decode(llr2)
    if not np.array_equal(u_sc, u_scl):
        ok = False
        if verbose:
            print("SCL L=1 与 SC 不一致")
    else:
        if verbose:
            print("SCL L=1 等价 SC 校验通过")

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    if not crc_check(enc, 8):
        ok = False
        if verbose:
            print("CRC-8 校验失败")
    else:
        if verbose:
            print("CRC-8 校验通过")

    if not ok:
        raise AssertionError("单元测试未通过")
    return True


if __name__ == "__main__":
    run_unit_tests()
