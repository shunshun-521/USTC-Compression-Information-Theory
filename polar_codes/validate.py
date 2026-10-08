"""模块数值正确性校验（各实验脚本开头调用）"""
import numpy as np

from channel import align_llr_for_decoder, awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_N = B_N F^{⊗n} 矩阵乘法一致：B @ G @ u = [1,0,1,1]
    expected = np.array([1, 0, 1, 1])
    if not np.array_equal(x, expected):
        raise AssertionError(f"编码器错误: got {x}, expected {expected}")

    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = 0.05
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_sent[info_idx] != u_hat[info_idx]):
            errors += 1
    if errors > 0:
        raise AssertionError(f"SC 无损校验失败: {errors}/100 帧错误")

    llr_test = np.array([1.0, -2.0, 3.0, -4.0])
    u_rec = sc_decode_recursive(llr_test, np.array([False, True, False, True]))
    u_nr = sc_decode(llr_test, np.array([False, True, False, True]))
    if not np.array_equal(u_rec, u_nr):
        raise AssertionError("递归与非递归 SC 不一致")

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_sent)
    llr = align_llr_for_decoder(compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8.0, 0.5)))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        raise AssertionError("L=1 SCL 与 SC 不一致")

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
