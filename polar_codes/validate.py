"""极化码模块单元测试与数值校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    """编码器：蝶形编码与生成矩阵一致。"""
    for N in (4, 8, 16, 64):
        rng = np.random.default_rng(0)
        for _ in range(20):
            u = rng.integers(0, 2, size=N)
            x1 = polar_encode(u)
            x2 = polar_encode_matrix(u)
            assert np.array_equal(x1, x2), f"N={N}: butterfly != matrix"

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_expected = polar_encode_matrix(u)
    assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"
    print(f"编码器校验通过 (N=4, u={u.tolist()} -> x={x.tolist()})")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([3, 5, 6, 7])
    assert np.array_equal(info, expected_info), f"GA N=8: {info}"
    assert len(frozen) == 4
    print("GA 构造校验通过")


def test_sc_lossless():
    """极低噪声下 SC 译码应无错误。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("SC 无损校验通过 (100/100)")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        r1 = sc_decode(llr, frozen_bits)
        r2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(r1, r2)
    print("递归/非递归 SC 一致性校验通过")


def test_scl_equiv_sc():
    """L=1 的 SCL 应等价于 SC。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(99)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL(L=1) ≡ SC 校验通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("CRC 校验通过")


def test_bp_noiseless():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])
    print(f"BP 低噪声校验通过 (iters={iters})")


def run_all():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    test_bp_noiseless()
    print("\n全部单元测试通过。")


if __name__ == "__main__":
    run_all()
