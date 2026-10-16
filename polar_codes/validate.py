"""模块数值校验（供实验脚本调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_match_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
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
            print(f"编码器与生成矩阵不一致: {x} vs {x_ref}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            sc_errors += 1
    if sc_errors > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比测试失败: {sc_errors}/100 帧错误")

    if not sc_decode_match_recursive(llr, frozen_bits):
        ok = False
        if verbose:
            print("递归与非递归 SC 不一致")

    frozen_bool = frozen_bits.astype(bool)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        ok = False
        if verbose:
            print("L=1 SCL 与 SC 不一致")

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    if not crc_check(coded, 8):
        ok = False
        if verbose:
            print("CRC 自洽性失败")

    if verbose:
        print("单元测试:", "通过" if ok else "失败")
    return ok


if __name__ == "__main__":
    run_unit_tests()
