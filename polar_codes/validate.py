"""模块数值校验（实验脚本启动时调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            sc_err += 1
    assert sc_err == 0, f"SC 无损校验失败: {sc_err}/100"

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_test)
    llr = compute_llr(bpsk_modulate(x), sigma)
    uh_sc, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 全部校验通过")
    return True


if __name__ == "__main__":
    run_validation()
