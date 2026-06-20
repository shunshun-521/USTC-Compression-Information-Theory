"""极化码模块单元测试（各实验脚本运行前调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x}, 期望 {x_ref}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info, expected_info), f"GA 构造错误: {info}"


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(12.0, K / N)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.default_rng().normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 无损译码失败"


def test_sc_recursive_matches():
    """递归 SC 在噪声less条件下应与非递归结果一致（抽样验证）。"""
    N = 16
    info_idx, _, _ = ga_construction(N, 8, 2.5)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    rng = np.random.default_rng(7)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, 8)
        llr = 100 * bpsk_modulate(polar_encode(u))
        u1 = sc_decode(llr, frozen_bool)
        u2 = sc_decode_recursive(llr, frozen_bool)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    llr = np.linspace(2.0, -2.0, N)
    u_sc = sc_decode(llr, frozen_bool)
    u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8), "CRC 校验失败"


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_recursive_matches()
    test_sc_lossless()
    test_scl_l1_equals_sc()
    test_crc()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all_tests()
