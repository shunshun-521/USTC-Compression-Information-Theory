"""极化码模块单元校验（供实验脚本调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import eb_n0_to_sigma, bpsk_modulate, awgn_channel, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validate(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits.astype(bool))
        assert np.array_equal(uh[info_idx], u[info_idx]), "SC 无损校验失败"

    frozen = frozen_bits.astype(bool)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc, _ = sc_decode(llr, frozen), None
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 等价"

    if verbose:
        print("validate.py: 全部校验通过")
    return True


if __name__ == "__main__":
    run_validate()
