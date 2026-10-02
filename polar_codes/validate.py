"""模块数值校验（各实验脚本启动时调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(u, sc_decode(llr, frozen)):
            sc_err += 1
    assert sc_err == 0, f"SC 校验失败: {sc_err}/100 帧错误"

    scl = SCLDecoder(N, frozen, list_size=1)
    l1_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh, _ = scl.decode(llr)
        if not np.array_equal(u, uh):
            l1_err += 1
    assert l1_err <= 2, f"L=1 SCL 应与 SC 相近，错误 {l1_err}/50"

    print("单元测试通过：编码器 / SC / SCL(L=1)")


if __name__ == "__main__":
    run_validations()
