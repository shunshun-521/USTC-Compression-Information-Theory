"""单元测试与模块验证"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, verify_sc_decoders
from decoder_scl import SCLDecoder, crc_encode, crc_check, verify_scl_equals_sc


def run_unit_tests():
    """运行所有单元测试"""
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first 20:", info256[:20])

    # SC 译码（12dB 下 100 帧无错误）
    verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=12.0)

    # SCL L=1 等价 SC
    verify_scl_equals_sc(N=64, K=32, num_frames=20)

    # CRC
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 0, 1, 1]), 8)
    assert crc_check(bits, 8)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
