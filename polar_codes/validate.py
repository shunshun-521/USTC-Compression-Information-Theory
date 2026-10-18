"""模块数值校验（在各 run_exp*.py 开头调用）。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = np.zeros((4, 4), dtype=int)
    for i in range(4):
        e = np.zeros(4, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与生成矩阵不一致: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u, u_hat), "SC 无损译码失败"

    frozen_i = frozen.astype(int)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen_i)
    u_scl, _ = SCLDecoder(N, frozen_i, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
