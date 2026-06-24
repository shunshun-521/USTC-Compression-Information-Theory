"""极化码模块单元测试。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行所有模块校验。"""
    # 编码器校验（生成矩阵一致性）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        codeword = polar_encode(u)
        symbols = bpsk_modulate(codeword)
        llr = compute_llr(symbols, eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 仍有 {errors} 帧错误"

    # L=1 SCL 等价于 SC
    scl = SCLDecoder(N, frozen_bool, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    codeword = polar_encode(u)
    llr = compute_llr(bpsk_modulate(codeword), eb_n0_to_sigma(5.0, K / N))
    u_sc = sc_decode(llr, frozen_bool)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
