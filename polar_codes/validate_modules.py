"""各模块数值校验（实验脚本开头调用）"""
import numpy as np

from encoder import polar_encode
from decoder_sc import sc_decode, verify_sc_lossless
from decoder_scl import verify_scl_equals_sc


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"

    assert verify_sc_lossless(N=64, K=32, num_frames=100, eb_n0_db=10.0), "SC 无损校验失败"
    assert verify_scl_equals_sc(N=64), "SCL(L=1) 与 SC 不一致"
    print("单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
