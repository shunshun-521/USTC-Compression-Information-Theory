"""单元测试：编码器与 SC/SCL 基本正确性。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits)
        if np.array_equal(uh, u):
            ok += 1
    assert ok >= 90, f"SC 高信噪比测试失败: {ok}/100"

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), sigma)
    uh_sc = sc_decode(llr, frozen_bits)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    print("verify_modules: 全部通过")


if __name__ == "__main__":
    run_all()
