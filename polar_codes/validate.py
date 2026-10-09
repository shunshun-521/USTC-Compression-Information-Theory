"""模块数值校验（仿真脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation():
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        assert np.array_equal(uh, u), "SC 高信噪比译码失败"

    def sc_wrap(llr):
        return sc_decode(llr, frozen), None

    def scl_wrap(llr):
        u_hat, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        return u_hat, None

    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
    llr = compute_llr(y, sigma)
    uh_sc, _ = sc_wrap(llr)
    uh_scl, _ = scl_wrap(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应与 SC 一致"

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
