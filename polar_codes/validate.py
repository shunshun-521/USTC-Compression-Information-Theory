"""模块数值校验（各实验脚本导入）"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.all(uh[info] == u[info]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"

    N = 128
    info, _, _ = ga_construction(N, 64, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, 64)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 等价"

    print("单元测试通过。")
