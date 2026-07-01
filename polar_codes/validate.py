"""共享单元测试，供各实验脚本调用"""
import numpy as np


def run_unit_tests():
    from encoder import polar_encode
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode, run_sc_validation_tests
    from decoder_scl import SCLDecoder, run_scl_validation_tests

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    run_sc_validation_tests()
    run_scl_validation_tests()
    print("All unit tests passed.")
