"""模块数值校验（仿真前运行）"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode_recursive, sc_decode_permuted
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u, apply_bit_reversal=False)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u, apply_bit_reversal=True)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode_permuted(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100 帧错误"

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int), True)), 0.2)
    u_sc = sc_decode_permuted(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    print("verify_modules: 全部通过")


if __name__ == "__main__":
    run_unit_tests()
