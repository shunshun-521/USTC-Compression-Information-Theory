"""单元测试：编码与译码一致性"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validate(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            err += 1
    assert err == 0, f"SC 高 SNR 校验失败: {err}/100 帧有错"

    u0 = np.zeros(N, dtype=int)
    u0[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u0)), 0.01)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 所有校验通过")
    return True


if __name__ == "__main__":
    run_validate()
