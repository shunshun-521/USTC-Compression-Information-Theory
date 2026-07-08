"""极化码模块单元测试"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128 first 20 info:", info256[:20])

    payload = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(payload, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在 10dB 下有 {sc_errors} 帧错误"

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), sigma)
    u_sc, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    u_scl = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
