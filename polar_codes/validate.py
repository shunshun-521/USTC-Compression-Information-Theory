"""模块数值校验（仿真脚本开头调用）"""
import numpy as np

from encoder import polar_encode, build_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: butterfly={x}, matrix={x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    a, b = sc_decode_recursive(llr, frozen_bits), sc_decode(llr, frozen_bits)
    assert np.array_equal(a, b), "递归与非递归 SC 不一致"

    sigma_mid = eb_n0_to_sigma(4.0, 0.5)
    y2 = awgn_channel(bpsk_modulate(x), sigma_mid, rng=rng)
    llr2 = compute_llr(y2, sigma_mid)
    u_sc = sc_decode(llr2, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr2)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
