"""极化码模块数值校验（各实验脚本开头调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_validations(verbose=True):
    N4 = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    u_rec = sc_decode(compute_llr(bpsk_modulate(x), 1e-3), np.zeros(N4, dtype=bool))
    assert np.array_equal(u_rec, u), f"N=4 无噪环回失败: u={u}, u_rec={u_rec}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    payload = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_scl, u_sc), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
