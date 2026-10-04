"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, arikan_generator
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = arikan_generator(4)
    assert np.array_equal(x, G @ u % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, len(info))
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, 0.5))
        uh = sc_decode(llr, frozen)
        if np.any(uh[info] != u[info]):
            err += 1
    assert err == 0, f"SC 无损校验失败: {err} 帧错误"

    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.0)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, len(info))
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    run_all()
