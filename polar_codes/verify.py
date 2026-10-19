"""模块数值校验（各实验脚本开头调用）。"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器：与 u @ G @ B_N 一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # SC 递归与非递归一致性
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    llr = np.random.default_rng(0).normal(0, 1, N)
    u1 = sc_decode_recursive(llr, frozen)
    u2 = sc_decode(llr, frozen)
    assert np.array_equal(u1, u2), "SC 递归/非递归不一致"

    # 高信噪比 SC 无损
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        payload = rng.integers(0, 2, K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)

    # L=1 SCL 等价 SC
    llr = np.random.default_rng(1).normal(0, 2, N)
    frozen_bits = frozen.astype(int)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("verify.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
