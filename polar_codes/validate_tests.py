"""极化码模块数值校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def run_validation_tests(verbose=True):
    """运行所有单元测试，失败时抛出 AssertionError。"""
    N4 = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_expected = u @ build_generator_matrix(N4) % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"
    if verbose:
        print(f"[PASS] 编码器: u={u} -> x={x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    noiseless_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != bits):
            noiseless_errors += 1
    assert noiseless_errors == 0, f"SC 无损译码失败: {noiseless_errors}/100 帧有错"
    if verbose:
        print("[PASS] SC 译码: 无损信道 100 帧零错误")

    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != bits):
            sc_errors += 1
    assert sc_errors <= 10, f"SC 高信噪比译码异常: {sc_errors}/100 帧有错"
    if verbose:
        print(f"[PASS] SC 译码: Eb/N0=10dB, {100 - sc_errors}/100 帧正确")

    # SCL L=1 等价于 SC
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        uh_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh_scl, uh_sc), "SCL(L=1) 与 SC 不一致"
    if verbose:
        print("[PASS] SCL(L=1) 等价于 SC")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print(f"[INFO] N=8, K=4: info={info8}, frozen={frozen8}")
        info256, _, _ = ga_construction(256, 128, 2.5)
        print(f"[INFO] N=256, K=128: first 20 info indices = {info256[:20]}")

    return True


if __name__ == "__main__":
    run_validation_tests()
