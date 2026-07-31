"""单元测试：验证各模块数值正确性。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4, f"编码器输出长度错误: {x}"

    # 无噪声编码-译码一致性
    for N in [4, 8, 16]:
        frozen = np.zeros(N, dtype=bool)
        for _ in range(20):
            msg = np.random.randint(0, 2, N)
            codeword = polar_encode(msg)
            llr = compute_llr(bpsk_modulate(codeword), 0.001)
            decoded = sc_decode(llr, frozen)
            assert np.array_equal(decoded, msg), f"N={N} 编码译码不一致"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], bits):
            errors += 1
    assert errors == 0, f"SC 无噪声测试失败: {errors}/100 帧错误"

    rng = np.random.default_rng(456)
    for _ in range(20):
        bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不等价"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
