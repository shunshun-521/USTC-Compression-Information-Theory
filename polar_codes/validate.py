"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma))
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            errs += 1
    assert errs == 0, f"SC 高信噪比测试失败: {errs} 帧错误"

    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = reorder_llr_for_decoder(compute_llr(bpsk_modulate(x), 1e-3))
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate: 所有单元测试通过")


if __name__ == "__main__":
    run_unit_tests()
