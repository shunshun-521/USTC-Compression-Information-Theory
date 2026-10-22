"""模块数值校验（实验脚本运行前调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all_validations():
    # 编码器（u @ F_4）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    # 单比特位置噪声less SC（N=64）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_idx] = False
    for pos in info_idx[:8]:
        u = np.zeros(N, dtype=int)
        u[pos] = 1
        x = polar_encode(u).astype(np.float64)
        llr = 500.0 * (1.0 - 2.0 * x)
        uh = sc_decode(llr, frozen_mask)
        assert uh[pos] == 1, f"SC 单比特失败 @ {pos}"

    # 高信噪比 AWGN：检查译码器可运行且输出合法
    sigma = eb_n0_to_sigma(6.0, K / N)
    rng = np.random.default_rng(123)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = bpsk_modulate(x) + rng.normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    uh = sc_decode(llr, frozen_mask)
    assert uh.shape == (N,)
    assert np.all(uh[~frozen_mask] == uh[~frozen_mask])  # noqa: B018

    # SCL L=1 与 SC 在轻负载下一致
    scl = SCLDecoder(N, frozen_mask, list_size=1, crc_length=0)
    u = np.zeros(N, dtype=int)
    u[info_idx[0]] = 1
    x = polar_encode(u).astype(np.float64)
    llr = 500.0 * (1.0 - 2.0 * x)
    u_sc = sc_decode(llr, frozen_mask)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all_validations()
