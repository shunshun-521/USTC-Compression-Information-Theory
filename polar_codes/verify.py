"""极化码模块单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """数值正确性校验。"""
    print("运行单元测试...")

    N4 = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]])
    G4 = np.kron(F, F)
    br4 = bit_reversal_permutation(N4)
    expected = np.dot(u, G4[br4]) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != u_sent[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在极低噪声下失败 {sc_errors}/100 帧"

    llr_test = compute_llr(bpsk_modulate(polar_encode(u_sent)), 1e-6)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_test)
    u_sc = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_scl, u_sc), "L=1 SCL 与 SC 不等价"

    print("单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
