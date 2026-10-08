"""极化码模块单元校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rev = bit_reversal_permutation(N)
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)[rev]
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    assert err == 0, f"SC 无损校验失败: {err}/100 帧错误"

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)[rev]
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应等价于 SC"

    if verbose:
        print("单元校验通过。")


if __name__ == "__main__":
    run_validation()
