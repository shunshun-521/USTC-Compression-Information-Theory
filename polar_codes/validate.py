"""
单元测试：验证编码器、SC/SCL 译码器、构造模块正确性。
在各实验脚本开头调用 run_all_tests()。
"""
import os
import numpy as np

FAST = os.environ.get("POLAR_FAST_SIM", "0") == "1"


def test_encoder():
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])  # u * F^{\otimes n}
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_lossless():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    trials = 20 if FAST else 100
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/{trials} 帧有误"


def test_scl_equals_sc():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode
    from decoder_scl import SCLDecoder

    N = 32
    K = 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"


def test_construction():
    from construction import ga_construction

    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8


def run_all_tests():
    test_encoder()
    test_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all_tests()
