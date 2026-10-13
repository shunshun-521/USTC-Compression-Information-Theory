"""模块数值校验（仿真脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, map_llr_to_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8,K=4 info:", info, "frozen:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = map_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), sigma))
        assert np.array_equal(sc_decode(llr, frozen_bits), u)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = map_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), sigma))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
