"""模块数值校验（实验脚本导入前调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errs += 1
    assert errs == 0, f"SC 无损校验失败: {errs}/100"

    # L=1 SCL 等价 SC
    rng = np.random.default_rng(456)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validation()
