"""模块数值校验（实验脚本导入）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen), u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 无损校验失败: {sc_errors}/100"

    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    scl_errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        uh, _ = scl.decode(llr)
        if not np.array_equal(uh, sc_decode(llr, frozen)):
            scl_errors += 1
    assert scl_errors == 0, f"SCL(L=1) 与 SC 不一致: {scl_errors}/20"

    if verbose:
        print("单元测试通过。")
