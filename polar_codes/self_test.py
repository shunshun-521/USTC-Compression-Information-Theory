"""极化码仿真实验公共单元测试"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, validate_sc_decoder
from decoder_scl import SCLDecoder, validate_scl_equals_sc


def run_unit_tests():
    """运行所有模块数值校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    assert validate_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=10.0), (
        "SC 译码校验失败"
    )
    assert validate_scl_equals_sc(N=64, K=32), "SCL L=1 与 SC 不等价"

    print("所有单元测试通过。")
