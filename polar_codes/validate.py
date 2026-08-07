"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # GA 构造校验
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7]), f"GA info 错误: {info}"
    assert np.array_equal(frozen, [0, 1, 2, 4]), f"GA frozen 错误: {frozen}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"

    # 路径度量校验：L=1 SCL 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(5.0, K / N)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不等价"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
