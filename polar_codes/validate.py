"""极化码模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造基本属性
    info_idx, frozen_idx, _ = ga_construction(8, 4, 2.5)
    assert len(info_idx) == 4 and len(frozen_idx) == 4
    assert len(np.intersect1d(info_idx, frozen_idx)) == 0

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        source = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = source
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat_sc = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat_sc, u_hat_rec)
        assert np.array_equal(u_hat_sc[info_idx], source)

    # L=1 SCL 等价 SC
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    frozen4 = np.array([False, False, True, True])
    u_sc = sc_decode(llr_test, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    print("All unit tests passed.")
