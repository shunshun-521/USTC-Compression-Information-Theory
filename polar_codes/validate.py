"""模块数值校验（在各实验脚本开头调用）。"""
import numpy as np
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, u @ G % 2), f"编码器与 G 不一致: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    err = 0
    for _ in range(100):
        msg = np.zeros(N, dtype=int)
        msg[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(msg)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(msg[info_idx], uh[info_idx]):
            err += 1
    assert err == 0, f"SC 高信噪比校验失败: {err}/100"

    msg2 = np.zeros(N, dtype=int)
    msg2[info_idx] = rng.integers(0, 2, K)
    llr_test = compute_llr(bpsk_modulate(polar_encode(msg2)), sigma)
    uh_sc = sc_decode(llr_test, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_test)
    assert np.array_equal(uh_sc[info_idx], uh_scl[info_idx]), "L=1 SCL 应等价 SC"

    print("validate.py: 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
