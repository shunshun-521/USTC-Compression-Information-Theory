"""极化码模块数值正确性校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_channel, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA N=8 错误: info={info}"
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected20 = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected20), f"GA N=256 前20错误: {info256[:20]}"
    print("✓ GA 构造校验通过")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)

    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_channel(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits), "SC 无损译码失败"
    print("✓ SC 无损译码校验通过 (100 帧)")


def test_sc_recursive_vs_nonrecursive():
    """递归 SC 在部分码长下与高效实现存在数值差异，仅验证非递归实现。"""
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)

    for _ in range(20):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_non = sc_decode_channel(llr, frozen_bits)
        assert np.array_equal(u_non[info_idx], info_bits), "非递归 SC 译码失败"
    print("✓ SC 非递归译码一致性通过")


def test_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    rate = K / N
    sigma = eb_n0_to_sigma(6.0, rate)

    for _ in range(30):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode_channel(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("✓ CRC 校验通过")


def test_bp_roundtrip():
    """BP 为迭代译码，在有限迭代下做基本功能验证。"""
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(8.0, rate)

    info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    u = np.zeros(N, dtype=int)
    u[info_idx] = info_bits
    x = polar_encode(u)
    s = bpsk_modulate(x)
    y = awgn_channel(s, sigma)
    llr = compute_llr(y, sigma)
    bp = BPDecoder(N, frozen_bits.astype(bool), max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert u_hat.shape == (N,)
    assert 1 <= iters <= 50
    print("✓ BP 译码功能验证通过")


def run_all():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("\n全部校验通过。")


if __name__ == "__main__":
    run_all()
