"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, channel_to_decoder_llr
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = channel_to_decoder_llr(compute_llr(bpsk_modulate(x), 0.01))
        uh = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(uh[info_idx], u[info_idx]))
    assert ok == 100, f"SC 无损校验失败: {ok}/100"

    sigma = eb_n0_to_sigma(10.0, K / N)
    ok2 = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = channel_to_decoder_llr(compute_llr(y, sigma))
        uh = sc_decode(llr, frozen_bits)
        ok2 += int(np.array_equal(uh[info_idx], u[info_idx]))
    assert ok2 >= 90, f"SC 高信噪比校验失败: {ok2}/100"

    u4 = np.array([1, 0, 1, 1])
    llr = channel_to_decoder_llr(compute_llr(bpsk_modulate(polar_encode(u4)), 0.01))
    uh_sc = sc_decode(llr, np.zeros(4, dtype=bool))
    assert np.array_equal(uh_sc, u4), "SC 译码错误"

    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(
        channel_to_decoder_llr(compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 0.01))
    )
    assert uh_scl is not None, "SCL 初始化失败"

    if verbose:
        print("[validate] 所有单元测试通过。")
