"""模块数值正确性校验（实验脚本启动前调用）。"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equivalent_to_sc


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(123)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N)
        )
        if np.array_equal(sc_decode(llr, fb), u):
            ok += 1
    assert ok == 100, f"SC 无损验证失败: {ok}/100"

    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(5.0, K / N))
    assert scl_equivalent_to_sc(N, fb, llr), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("单元测试全部通过。")
