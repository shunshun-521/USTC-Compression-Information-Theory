"""模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all_validations():
    """运行全部单元测试，失败时抛出 AssertionError"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4, f"编码器输出长度错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), u2), "编码器自洽性检验失败"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在极低噪声下失败 {errors}/100 帧"

    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应与 SC 等价"

    print("全部校验通过。")


if __name__ == "__main__":
    run_all_validations()
