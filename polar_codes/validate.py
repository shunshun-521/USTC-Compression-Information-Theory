"""数值正确性校验。"""
import numpy as np
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(0)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.all(uh[info] == u[info]):
            errs += 1
    assert errs == 0, f"SC @10dB 失败帧数: {errs}"

    scl = SCLDecoder(N, frozen, list_size=1)
    uh1, _ = scl.decode(llr)
    uh0, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh1, uh0), "SCL L=1 不稳定"

    print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_unit_tests()
