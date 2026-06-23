"""单元测试：验证各模块数值正确性"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, verify_sc_lossless
from decoder_scl import SCLDecoder, crc_check, crc_encode, verify_scl_equals_sc
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"


def test_sc_lossless():
    assert verify_sc_lossless(N=64, K=32, num_frames=100, eb_n0_db=10.0), (
        "SC 译码在高 SNR 下应无错误"
    )


def test_scl_equals_sc():
    assert verify_scl_equals_sc(N=64, K=32), "L=1 时 SCL 应等价于 SC"


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)


def run_all():
    test_encoder()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_all()
