"""极化码模块数值校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行编码器、SC、SCL 单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    if verbose:
        print("编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = 0.01
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比校验失败: {errors}/100 帧错误"
    if verbose:
        print("SC 译码校验通过 (N=64, Eb/N0=10dB, 100帧)")

    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_scl, u_sc), "L=1 时 SCL 应等价于 SC"
    if verbose:
        print("SCL 路径度量校验通过 (L=1 等价 SC)")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print(f"GA 构造 N=8: info={info8}, frozen={frozen8}")
        info256, _, _ = ga_construction(256, 128, 2.5)
        print(f"GA 构造 N=256 前20个 info: {info256[:20]}")

    return True


if __name__ == "__main__":
    run_unit_tests()
