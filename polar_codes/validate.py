"""极化码模块单元测试与校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, prepare_channel_llr
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    # 编码器校验（标准极化码：u=[1,0,1,1] -> x=[1,0,1,1]）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # 自洽性：u=[0,0,1,1] -> x=[0,0,1,1]
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1])

    # CRC 校验
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = prepare_channel_llr(
            compute_llr(bpsk_modulate(polar_encode(u)), sigma), N
        )
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下出现 {errors} 帧错误"

    # L=1 SCL 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = prepare_channel_llr(compute_llr(bpsk_modulate(polar_encode(u)), sigma), N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应与 SC 一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
