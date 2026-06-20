"""单元测试：各实验脚本运行前调用"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_generator_matrix, verify_encoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    ok, _, x_mat = verify_encoder(u)
    assert ok, f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info256) == 128


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)

    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(5.0, K / N))
        u_sc = sc_decode(llr, fb)
        u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)


def run_all_tests():
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all_tests()
