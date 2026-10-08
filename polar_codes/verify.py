"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for t in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, np.random.default_rng(t))
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if np.all(uh[info] == u[info]):
            ok += 1
    assert ok == 100, f"SC 10dB 校验失败: {ok}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_scl, uh), "L=1 SCL 应与 SC 一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
