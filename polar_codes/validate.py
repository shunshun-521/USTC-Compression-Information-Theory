"""单元测试：编码器、SC/SCL、路径度量"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_codeword
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False

    for _ in range(100):
        payload = np.random.randint(0, 2, K)
        u = build_codeword(payload, info, N)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + np.random.randn(N) * sigma, sigma)
        assert np.array_equal(sc_decode(llr, frozen_bits), u)

    payload = np.random.randint(0, 2, K)
    u = build_codeword(payload, info, N)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
    uh_sc = sc_decode(llr, frozen_bits)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 全部通过")
    return True


if __name__ == "__main__":
    run_validation()
