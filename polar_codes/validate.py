"""模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import crc_encode, crc_check


def run_unit_tests():
    # 编码器：与 G_N 一致
    u = np.array([1, 0, 1, 1])
    G = build_generator_matrix(4)
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    # SC：N=4,K=2 高信噪比
    N, K = 4, 2
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            err += 1
    assert err <= 5, f"SC 校验失败: {err}/100 帧错误"

    payload = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(payload, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
