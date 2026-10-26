"""模块数值校验（实验脚本开头调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    # u @ G_N = [0,0,1,1]（蝶形编码，无额外倒序）
    u = np.array([0, 1, 0, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errs = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        # 无噪信道（用接收符号作软信息，与 SC 树译码一致）
        y = bpsk_modulate(polar_encode(u))
        u_hat = sc_decode(y, frozen_bits)
        if np.any(u_hat[info_idx] != info):
            errs += 1
    assert errs == 0, f"SC 高信噪比测试失败: {errs}/100 帧错误"

    # L=1 SCL ~ SC
    y = bpsk_modulate(polar_encode(u))
    fb4 = np.array([0, 1, 1, 1], dtype=int)
    u_sc = sc_decode(y, fb4)
    u_scl, _ = SCLDecoder(4, fb4, list_size=1).decode(y)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    if verbose:
        print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
