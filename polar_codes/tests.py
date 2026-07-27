"""极化码模块单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode


def run_unit_tests():
    """模块正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, polar_encode_matrix(u)), f"编码器与矩阵不一致: {x}"
    uh = sc_decode(100.0 * bpsk_modulate(x), np.zeros(4, dtype=bool))
    assert np.array_equal(uh, u), f"编码-译码往返失败: {uh}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        if not np.array_equal(u[info_idx], sc_decode(llr, frozen_bits)[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 仍有 {errors} 帧错误"

    from decoder_scl import SCLDecoder
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(6.0, K / N))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"
    print("单元测试全部通过。")


if __name__ == "__main__":
    run_unit_tests()
