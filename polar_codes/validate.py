"""模块数值校验（各 run_exp*.py 开头调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import llr_to_sc_domain


def run_validation(verbose=True):
    # 编码器：蝶形 + 比特倒序（与 GA/译码流程一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8 info:", info, "frozen:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("GA N=256 info first 20:", info256[:20])

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = llr_to_sc_domain(compute_llr(bpsk_modulate(x), sigma), N)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 校验失败: {errors}/100 帧错误"

    # SCL L=1 等价 SC
    u_pad = np.zeros(N, dtype=int)
    u_pad[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u_pad)
    llr = llr_to_sc_domain(compute_llr(bpsk_modulate(x), sigma), N)
    fb = frozen_bits.astype(bool)
    uh_sc = sc_decode(llr, fb)
    uh_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "SCL(L=1) 与 SC 不一致"

    if verbose:
        print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_validation()
