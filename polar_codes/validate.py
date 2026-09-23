"""模块正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def validate_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 译码在高信噪比下失败"
    print("SC 译码校验通过")


def validate_scl_equivalence():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"
    print("SCL 路径度量校验通过")


def validate_crc():
    bits = np.random.default_rng(0).integers(0, 2, 32)
    for length in (8, 16):
        encoded = crc_encode(bits, length)
        assert crc_check(encoded, length), f"CRC-{length} 校验失败"
    print("CRC 校验通过")


def validate_bp_decoder():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(9)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 80.0, -80.0)
        u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
        assert np.array_equal(u_hat, u), "BP 译码在无损条件下失败"
    print("BP 译码校验通过")


def run_all_validations():
    print("=" * 50)
    print("运行模块校验...")
    validate_encoder()
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, info前20: {info256[:20]}")
    validate_sc_decoder()
    validate_scl_equivalence()
    validate_crc()
    validate_bp_decoder()
    print("所有校验通过")
    print("=" * 50)


if __name__ == "__main__":
    run_all_validations()
