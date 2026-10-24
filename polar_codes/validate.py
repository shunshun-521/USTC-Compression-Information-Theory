"""极化码模块单元测试（仿真脚本运行前调用）"""
import os
import numpy as np

from encoder import polar_encode, polar_encode_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器与矩阵不一致: {x} vs {x_ref}"
    u2 = np.array([0, 1, 0, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器错误: {polar_encode(u2)}"

    # CRC
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)

    # SC 无损
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    err = 0
    frames = 100 if os.environ.get("POLAR_QUICK") != "1" else 30
    for _ in range(frames):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(1 - 2 * x, 0.001)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            err += 1
    assert err == 0, f"SC 无损测试失败 {err}/{frames} 帧"

    # L=1 SCL 等价 SC
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(1 - 2 * x, 0.001)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("validate.py: 所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
