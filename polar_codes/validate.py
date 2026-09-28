"""模块数值校验（各实验脚本可调用）。"""
import os
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_validation(verbose=True):
    if verbose:
        print("Running polar_codes validation...")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    msg = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    with_crc = crc_encode(msg, 8)
    assert crc_check(with_crc, 8)

    if verbose:
        print("Validation passed.")


if __name__ == "__main__":
    if os.environ.get("POLAR_SKIP_VALIDATE", "0") != "1":
        run_validation()
