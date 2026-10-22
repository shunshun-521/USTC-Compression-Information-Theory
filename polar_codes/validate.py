"""模块数值校验（实验脚本开头调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, reorder_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    ga_construction(8, 4, 2.5)

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = reorder_llr_for_decoder(compute_llr(bpsk_modulate(x), 0.05))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    if verbose:
        print("validate.py: 所有校验通过")
    return True


if __name__ == "__main__":
    run_validation()
