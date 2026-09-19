"""极化码模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("✓ 编码器校验通过")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(np.intersect1d(info, frozen)) == 0
    print(f"✓ GA 构造 N=8: info={info}, frozen={frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"✓ GA 构造 N=256 前20: {info256[:20]}")


def test_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    errors = 0

    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int32)
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_hat_nr = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)

        assert np.array_equal(u_hat_nr, u_hat_rec), "递归与非递归 SC 不一致"

        if not np.array_equal(u_hat_nr[info_idx], payload):
            errors += 1

    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors}/100 错误"
    print("✓ SC 译码校验通过（递归=非递归，100帧无错）")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(5.0, rate)

    for _ in range(20):
        payload = rng.integers(0, 2, size=K, dtype=np.int32)
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    print("✓ SCL L=1 等价 SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    print("✓ CRC 校验通过")


if __name__ == '__main__':
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equiv_sc()
    test_crc()
    print("\n所有校验通过！")
