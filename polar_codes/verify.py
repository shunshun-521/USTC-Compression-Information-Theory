"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G4 = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G4) % 2), f"编码器错误: {x}"
    u_ex = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u_ex), (u_ex @ G4) % 2), "编码器手算样例错误"

    N = 4
    G = polar_generator_matrix(N)
    for _ in range(20):
        u_rand = np.random.randint(0, 2, N)
        x_enc = polar_encode(u_rand)
        x_mat = (u_rand @ G) % 2
        assert np.array_equal(x_enc, x_mat), "编码与生成矩阵不一致"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])

    llr_test = np.array([1.0, -2.0, 0.5, 3.0])
    u_rec = sc_decode_recursive(llr_test, np.array([False, True, False, True]))
    u_nr = sc_decode(llr_test, np.array([0, 1, 0, 1]))
    assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 不一致"

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    llr = np.array([2.0, -1.0, 0.3, -0.2, 1.1, -3.0, 0.0, 4.0] * 4, dtype=float)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    print("All polar_codes unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
