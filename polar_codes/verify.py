"""单元测试：编码器与 SC/SCL 译码正确性。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, align_llr_for_decoder
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器校验（G_N 约定：x = u @ G_N）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # SC 无损校验 N=64, K=32
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh[info_idx], u[info_idx])

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
    uh_sc = sc_decode_recursive(align_llr_for_decoder(llr), frozen_bits)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl)

    print("verify.py: 全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
