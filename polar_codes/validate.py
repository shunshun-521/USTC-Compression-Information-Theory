"""模块数值校验（实验脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, generator_matrix, polar_encode_with_bit_reversal
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_match_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def run_all_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_br = polar_encode_with_bit_reversal(u)
    x_ref = (u @ generator_matrix(4)) % 2
    assert np.array_equal(x_br, x_ref), f"编码器与 G 矩阵不一致: {x_br} vs {x_ref}"
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"蝶形编码 N=4: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        llr = rng.normal(0, 2, N)
        a, b = sc_decode_match_recursive(llr, frozen_bits)
        assert np.array_equal(a, b), "SC 递归与非递归不一致"

    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u_full = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u_full[info_idx] = bits
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat[info_idx], bits):
            ok += 1
    assert ok == 100, f"SC 高 SNR 测试失败: {ok}/100"

    N4 = 4
    fb4 = np.array([0, 1, 1, 1], dtype=int)
    u4 = np.array([1, 0, 0, 0], dtype=int)
    llr4 = compute_llr(bpsk_modulate(polar_encode(u4)), 0.5)
    u_sc = sc_decode(llr4, fb4)
    u_scl, _ = SCLDecoder(N4, fb4, list_size=1).decode(llr4)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    info = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)

    bp = BPDecoder(N, frozen_bits, max_iter=8)
    llr_bp = compute_llr(y, sigma)
    u_hat, _ = bp.decode(llr_bp)
    assert u_hat.shape == (N,)

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
