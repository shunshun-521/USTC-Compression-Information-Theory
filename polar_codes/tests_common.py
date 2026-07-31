"""极化码仿真公共单元测试"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    # 编码器校验（G_N = B_N F^{⊗n}）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # SC 译码校验（无损）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(20.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    # 路径度量：L=1 SCL 等价于 SC
    llr_test = rng.normal(0, 1, N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    assert np.array_equal(u_scl, sc_decode(llr_test, frozen_bits))

    # CRC 自洽
    bits = rng.integers(0, 2, 20)
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)

    print("All unit tests passed.")
