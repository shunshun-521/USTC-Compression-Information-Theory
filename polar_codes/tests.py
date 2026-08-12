"""单元测试与数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests(verbose=True):
    """运行所有模块单元测试，失败则抛出 AssertionError。"""
    # 编码器：蝶形与矩阵乘法一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器蝶形/矩阵不一致: {x} vs {xm}"

    # SC 译码：高信噪比下应无错误
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 高 SNR 译码失败"

    # SCL L=1 等价于 SC
    frozen_bool = frozen_bits.astype(bool)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_sc = sc_decode(llr, frozen_bool)
    u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"

    if verbose:
        print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
