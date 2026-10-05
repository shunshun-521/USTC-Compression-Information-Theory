"""模块数值校验（实验脚本导入）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validations(verbose=True):
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        assert np.array_equal(sc_decode(llr, frozen_bits), u)

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), sigma)
    assert np.array_equal(SCLDecoder(N, frozen_bits, list_size=1).decode(llr)[0], sc_decode(llr, frozen_bits))
    assert np.array_equal(SCLDecoder(N, frozen_bits, list_size=4).decode(llr)[0], sc_decode(llr, frozen_bits))

    if verbose:
        print("validations: OK")


if __name__ == "__main__":
    run_validations()
