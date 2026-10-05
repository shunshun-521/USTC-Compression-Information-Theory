"""模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_all_validations():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # SC 无损（递归 vs 非递归）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode(llr, frozen)
        u2 = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u1, u2)
        assert np.array_equal(u1[info_idx], payload)

    # SCL L=1 等价 SC
    u4 = np.array([1, 0, 1, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u4)), 0.01)
    frozen4 = np.array([False, False, False, False])
    u_sc = sc_decode(llr, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)

    print("All validations passed.")


if __name__ == "__main__":
    run_all_validations()
