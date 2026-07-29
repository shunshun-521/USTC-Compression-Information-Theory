"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def _build_generator_matrix(N):
    """构造 G_N = B_N * F^{⊗n}。"""
    F = np.array([[1, 0], [1, 1]])
    G = np.array([[1]])
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    brp = bit_reversal_permutation(N)
    return G[brp, :]


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    # 编码器校验（生成矩阵验证）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = _build_generator_matrix(4)
    x_expected = u @ G % 2
    assert np.array_equal(x, x_expected), f"编码器错误: {x}, 期望 {x_expected}"
    print("[PASS] 编码器校验 (G 矩阵)")

    # GA 构造校验
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info_indices: {info}, frozen_indices: {frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 前20个 info_indices: {info256[:20]}")

    # SC 译码校验（无损验证）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 帧错误"
    print("[PASS] SC 译码无损校验 (N=64, 100帧)")

    # 递归与非递归 SC 一致性
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("[PASS] 递归/非递归 SC 一致性")

    # 路径度量校验：L=1 SCL 等价于 SC
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] L=1 SCL 等价于 SC")

    # CRC 校验
    info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info_bits, 8)
    assert crc_check(coded, 8), "CRC 编码/校验失败"
    print("[PASS] CRC 编码/校验")

    print("=" * 50)
    print("所有单元测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_unit_tests()
