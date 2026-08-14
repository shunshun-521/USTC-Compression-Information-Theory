"""极化码模块单元测试与校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    # 编码器校验：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_generator_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"

    # SC 译码校验（极低噪声下应基本无错）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    assert sc_errors <= 5, f"SC 译码在 Eb/N0=10dB 错误过多: {sc_errors}/100"

    # 路径度量校验：L=1 的 SCL 应等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    scl_mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(5.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_mismatch += 1
    assert scl_mismatch == 0, f"SCL(L=1) 与 SC 不一致: {scl_mismatch}/50"

    # CRC 基本校验
    info = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    payload = crc_encode(info, 8)
    assert crc_check(payload, 8)
    assert len(payload) == len(info) + 8

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
