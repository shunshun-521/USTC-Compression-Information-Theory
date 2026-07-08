"""单元测试：验证各模块正确性"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    if verbose:
        print("[PASS] 编码器校验 N=4")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码噪声less失败: {errors}/100"
    if verbose:
        print("[PASS] SC 译码噪声less N=64, 100 帧")

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 0.001)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"
    if verbose:
        print("[PASS] SCL L=1 等价 SC")

    info8, _, _ = ga_construction(8, 4, 2.5)
    print("\nN=8, K=4, Eb/N0=2.5dB 构造验证:")
    print("info_indices:", info8)
    print("frozen_indices:", np.where(np.isin(np.arange(8), info8, invert=True))[0])

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, 前 20 个 info_indices:", info256[:20])


if __name__ == "__main__":
    run_unit_tests()
