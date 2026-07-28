"""单元测试：各模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError"""
    # 编码器：蝶形结构自洽性（N=4 手算 x=[1,0,1,1]）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert x.shape == (4,), f"编码输出形状错误: {x.shape}"

    # 编码-译码无损（N=4 穷举）
    for i in range(16):
        u_test = np.array([(i >> j) & 1 for j in range(4)])
        llr = 100.0 * (1.0 - 2.0 * polar_encode(u_test))
        u_hat = sc_decode(llr, np.zeros(4, dtype=bool))
        assert np.array_equal(u_hat, u_test), f"N=4 译码失败: u={u_test}"

    # SC 高信噪比测试
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec)
        assert np.array_equal(u_hat[info_idx], u_sent[info_idx])

    # SCL L=1 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)

    # CRC
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
