"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            err += 1
    assert err == 0, f"SC 高信噪比校验失败: {err}/100 帧错误"

    u0 = np.zeros(N, dtype=int)
    u0[info_idx] = rng.integers(0, 2, K)
    x0 = polar_encode(u0)
    llr0 = compute_llr(bpsk_modulate(x0), sigma)
    uh_sc = sc_decode(llr0, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr0)
    assert np.array_equal(uh_sc[info_idx], uh_scl[info_idx]), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 所有校验通过。")
    return True


if __name__ == "__main__":
    run_validation()
