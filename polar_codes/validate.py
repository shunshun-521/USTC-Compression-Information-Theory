"""单元测试与数值校验"""
import numpy as np

from channel import bpsk_modulate, compute_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行所有模块单元测试，失败则抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info8, [0, 3, 5, 6]), f"GA N=8 错误: {info8}"

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"GA N=256 前20个信息位索引: {info256[:20]}")

    # SC 无损校验
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), "SC 无损译码验证失败"

    # 递归与非递归 SC 一致性
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    uh1 = sc_decode(llr_test, np.zeros(4, dtype=int))
    uh2 = sc_decode_recursive(llr_test, np.zeros(4, dtype=int))
    assert np.array_equal(uh1, uh2), "递归/非递归 SC 不一致"

    # SCL L=1 等价 SC
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    sc_out = sc_decode(llr, frozen_bits)
    scl_out, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(sc_out, scl_out), "SCL L=1 与 SC 不等价"

    # BP 冒烟测试
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    bp_out, _ = bp.decode(llr)
    assert len(bp_out) == N

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
