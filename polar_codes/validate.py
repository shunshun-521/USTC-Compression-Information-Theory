"""模块数值校验（实验前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import compute_llr, channel_llr_for_decoder, eb_n0_to_sigma, bpsk_modulate, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    info, _, _ = ga_construction(8, 4, 2.5)
    print("N=8,K=4 info:", info)

    info256, frozen256, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    err = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = channel_llr_for_decoder(compute_llr(y, sigma))
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh, u):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100"

    llr0 = channel_llr_for_decoder(compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), sigma))
    u0, _ = SCLDecoder(N, frozen, list_size=1).decode(llr0)
    u0_sc = sc_decode(llr0, frozen)
    assert np.array_equal(u0, u0_sc), "L=1 SCL 应与 SC 一致"

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validations()
