"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decoder,
)
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    # 编码器：自逆性（G 为自逆矩阵）
    u4 = np.array([1, 0, 1, 1])
    x4 = polar_encode(u4)
    assert np.array_equal(polar_encode(x4), u4), f"编码器自逆性失败: {x4}"

    # 构造 sanity
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4

    # SC 高信噪比无损
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = reorder_llr_for_decoder(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng=rng), sigma)
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 译码在高信噪比下失败"

    # SCL L=1 等价 SC
    llr = reorder_llr_for_decoder(np.linspace(5, -5, N))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"

    # CRC
    info_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    coded = crc_encode(info_bits, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.concatenate([info_bits, np.zeros(8, int)]), 8)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
