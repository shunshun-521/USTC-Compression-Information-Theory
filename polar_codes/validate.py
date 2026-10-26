"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_decoder_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    info, _, _ = ga_construction(8, 4, 2.5)
    print("N=8,K=4 info:", info)
    info256, frozen256, _ = ga_construction(256, 128, 2.5)
    print("N=256,K=128 first 20 info:", info256[:20])
    print("N=256 frozen first 20:", frozen256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        assert np.array_equal(sc_decode(llr, frozen), u)

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = prepare_decoder_llr(compute_llr(bpsk_modulate(polar_encode(u)), sigma), N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validation()
