"""
单元测试与公共校验函数
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def run_unit_tests():
    """运行所有模块的数值正确性校验。"""
    # 编码器：与生成矩阵一致
    G = build_generator_matrix(4)
    for u in (
        np.array([1, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
    ):
        x = polar_encode(u)
        x_ref = (u @ G) % 2
        assert np.array_equal(x, x_ref), f"编码器错误: u={u}, x={x}, ref={x_ref}"

    # SC 译码：极低噪声下应无错
    N, K = 64, 32
    design_eb_n0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb_n0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info_bits)

    # 递归与非递归 SC 一致
    llr_test = rng.normal(0, 2, size=N)
    u_rec = sc_decode_recursive(llr_test, frozen_bits)
    u_nonrec = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_rec, u_nonrec)

    # L=1 SCL 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    assert np.array_equal(u_scl, u_nonrec)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
