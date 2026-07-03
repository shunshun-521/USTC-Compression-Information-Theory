"""单元测试与数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    """运行所有模块校验，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [3, 5, 6, 7]), f"GA N=8 构造异常: {info8}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_first20 = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected_first20), (
        f"GA N=256 前20位不匹配: {info256[:20]}"
    )

    # SC 递归与非递归一致性
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_rec = sc_decode(llr, frozen_bits)
        u_ref = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_ref), "SC 递归与非递归不一致"
        assert np.array_equal(u_rec, u), "SC 无损译码失败"

    # 低噪声 SC 校验
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for trial in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], bits):
            errors += 1
    assert errors == 0, f"低噪声 SC 校验失败: {errors}/100 帧错误"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.05)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    if verbose:
        print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
