"""模块数值校验（实验脚本启动时调用）。"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validate(verbose=True):
    if os.environ.get("POLAR_SKIP_VALIDATE", "").lower() in ("1", "true", "yes"):
        return

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 校验失败: {errors}/100 帧有错"

    scl = SCLDecoder(N, frozen, list_size=1)
    mism = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mism += 1
    assert mism == 0, f"SCL(L=1) 与 SC 不一致: {mism}/{20}"

    if verbose:
        print("validate: 全部通过")


if __name__ == "__main__":
    run_validate()
    print("OK")
