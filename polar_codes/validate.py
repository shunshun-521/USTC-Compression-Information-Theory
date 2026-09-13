"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode
from simulation import run_simulation


def validate_all(verbose=True):
    """运行全部校验，失败时抛出 AssertionError。"""
    # 编码器校验（G_N = B_N F^{⊗n}，比特倒序由 PSCD 译码顺序等价实现）
    u = np.array([1, 1, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 0, 1]), f"编码器错误: {x}"

    # GA 构造 sanity check
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=8 info:", info8, "frozen:", frozen8)
        print("N=256 first 20 info:", info256[:20])

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u, u_hat), "SC 译码在高 SNR 下失败"

    # SCL L=1 等价 SC
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 与 SC 不一致"

    # CRC
    payload = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(payload, 8)

    # BP 冒烟测试
    bp = BPDecoder(N, frozen, max_iter=20)
    u_hat, _ = bp.decode(llr)
    assert u_hat.shape == (N,)

    if verbose:
        print("全部单元测试通过。")


if __name__ == "__main__":
    validate_all()
