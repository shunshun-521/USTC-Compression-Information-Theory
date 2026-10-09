"""模块数值校验（供实验脚本调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器：与 G_N 矩阵一致
    N = 4
    G = build_generator_matrix(N)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
    # 与 G_N 一致：u=[1,0,1,1] -> x=[1,0,1,1]
    assert np.array_equal(polar_encode(np.array([1, 0, 1, 1])), [1, 0, 1, 1])

    # 递归与非递归 SC 一致
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    llr = np.random.default_rng(0).normal(0, 1, N)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1, u2), "SC 递归/非递归不一致"

    # 高 SNR：短码 N=8 上统计 BLER 应明显低于 0.5
    from channel import decoder_llr

    Ns, Ks = 8, 4
    info_s, _, _ = ga_construction(Ns, Ks, 2.5)
    fb_s = np.ones(Ns, dtype=bool)
    fb_s[info_s] = False
    sigma = eb_n0_to_sigma(6.0, Ks / Ns)
    rng = np.random.default_rng(123)
    errs = 0
    frames = 200
    for _ in range(frames):
        payload = rng.integers(0, 2, Ks)
        u = np.zeros(Ns, dtype=int)
        u[info_s] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        u_hat = sc_decode(decoder_llr(y, sigma), fb_s)
        if not np.array_equal(u_hat[info_s], payload):
            errs += 1
    assert errs / frames <= 0.36, f"高 SNR SC 统计 BLER 异常: {errs/frames:.3f}"

    # L=1 SCL 等价 SC（N=64）
    payload = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
    llr64 = decoder_llr(y, sigma)
    u_sc = sc_decode(llr64, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr64)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
