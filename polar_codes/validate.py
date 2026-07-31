"""极化码仿真实验共享校验与配置。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    """编码器、SC/SCL 基础校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100"

    uh, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(
        compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
    )
    assert np.array_equal(uh, u), "L=1 SCL 应等价于 SC"

    print("单元测试通过。")
