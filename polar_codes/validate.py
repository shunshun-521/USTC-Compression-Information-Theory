"""极化码模块单元测试。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    # SC 无损译码
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        assert np.array_equal(u, sc_decode(llr, frozen_bits))
        assert np.array_equal(u, sc_decode_recursive(llr, frozen_bits))

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(5.0, 0.5))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl)

    # CRC
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(coded[:-1], 8)

    # BP 基本功能（短码长 N=8）
    N8, K8 = 8, 4
    info8, _, _ = ga_construction(N8, K8, 2.5)
    fb8 = np.ones(N8, dtype=int)
    fb8[info8] = 0
    bp8 = BPDecoder(N8, fb8, max_iter=100)
    bp_ok = 0
    for _ in range(16):
        u8 = np.zeros(N8, dtype=int)
        u8[info8] = rng.integers(0, 2, K8)
        llr8 = compute_llr(bpsk_modulate(polar_encode(u8)), eb_n0_to_sigma(10.0, 0.5))
        u8_bp, _ = bp8.decode(llr8)
        bp_ok += np.array_equal(u8, u8_bp)
    assert bp_ok > 0, "BP 译码器在 N=8 高信噪比下无正确帧"

    if verbose:
        print("全部单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
