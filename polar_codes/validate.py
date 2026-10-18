"""模块数值校验（各实验脚本启动时调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u.astype(np.int64) @ generator_matrix(4)) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info]), "SC 译码错误"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=np.int8)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl[info], u[info]), "SCL(L=1) 应与 SC 一致"
