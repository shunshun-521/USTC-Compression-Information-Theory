"""极化码模块单元测试（各实验脚本启动时调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, arikan_generator
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = arikan_generator(4)
    assert np.array_equal(x, (G @ u) % 2), f"编码器错误: {x}"

    # SC 高信噪比
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u)

    # SCL L=1 等价 SC
    u4 = np.array([1, 0, 1, 1])
    llr4 = compute_llr(bpsk_modulate(polar_encode(u4)), 0.1)
    frozen4 = np.zeros(4, dtype=bool)
    uh_sc = sc_decode(llr4, frozen4)
    uh_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr4)
    assert np.array_equal(uh_sc, uh_scl)

    print("[verify] 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
