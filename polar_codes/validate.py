"""模块数值校验（实验脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh, u), "SC 高信噪比译码失败"

    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(
        compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    )
    assert np.array_equal(uh_scl, u), "SCL(L=1) 应与 SC 一致"

    if verbose:
        print("validate.py: 全部校验通过")
    return True


if __name__ == "__main__":
    run_validation()
