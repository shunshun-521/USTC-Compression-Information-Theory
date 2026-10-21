"""模块数值校验（实验脚本开头调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8 info:", info8, "frozen:", frozen8)

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    frozen_b = frozen.astype(bool)
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        if not np.array_equal(sc_decode(llr, frozen_b), u):
            sc_err += 1
    assert sc_err == 0, f"SC 高信噪比测试失败: {sc_err}/100"

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen_b, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
    uh_sc = sc_decode(llr, frozen_b)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("validate.py: 所有校验通过")
    return True


if __name__ == "__main__":
    run_validation()
