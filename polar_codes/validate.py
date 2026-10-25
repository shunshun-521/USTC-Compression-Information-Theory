"""模块数值校验（仿真脚本启动前调用）"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if np.array_equal(sc_decode(llr, frozen), u):
            ok += 1
    assert ok == 100, f"SC 高信噪比校验失败: {ok}/100"

    u_hat, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_hat, u), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
