"""
蒙特卡洛仿真单元测试（各实验脚本共用）
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    """运行编码器、SC/SCL/CRC 校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng=rng),
            sigma,
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u_sent[info_idx])

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng=rng),
            sigma,
        )
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen)
        assert np.array_equal(u_scl, u_sc)

    info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
