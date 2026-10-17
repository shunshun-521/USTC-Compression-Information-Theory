"""模块数值校验（各实验脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    frozen_bool = frozen.astype(bool)
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        uh = sc_decode(llr, frozen_bool)
        assert np.array_equal(uh[info], u[info]), "SC 高信噪比译码失败"

    # L=1 SCL 等价 SC
    u_full = np.zeros(N, dtype=int)
    u_full[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_full)), 0.3)
    uh_sc = sc_decode(llr, frozen_bool)
    uh_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
