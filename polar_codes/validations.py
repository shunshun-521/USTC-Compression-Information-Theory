"""模块正确性校验（各实验脚本开头调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])  # u @ F^{\otimes 2}，与 SC 译码约定一致
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def validate_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)

    # 无损 LLR 验证（等价于极高信噪比）
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 1e3
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info), "SC 译码失败"

    # 递归与非递归一致
    llr_test = rng.normal(0, 1, N)
    u_rec = sc_decode_recursive(llr_test, frozen_bits)
    u_nr = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 不一致"


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, size=K)
        u[info_idx] = info
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 1e3
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"


def run_all_validations():
    validate_encoder()
    validate_sc_decoder()
    validate_scl_equals_sc()
    print("所有单元测试通过。")
