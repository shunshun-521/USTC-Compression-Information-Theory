"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_N = B_N F^{⊗n} 一致（规格中 [0,0,1,1] 与常见定义不一致）
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
    # 规格中的 [0,0,1,1] 与 G_N=B_N F^{⊗n} 定义不一致


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"


def validate_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        a = sc_decode(llr, frozen_bits)
        b = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(a, b)


def validate_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)


def run_all():
    validate_encoder()
    validate_crc()
    validate_sc_recursive_match()
    validate_sc_noiseless()
    validate_scl_l1_equals_sc()
    print("All validations passed.")


if __name__ == "__main__":
    run_all()
