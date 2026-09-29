"""模块数值正确性校验（在各 run_exp*.py 开头调用）。"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if np.array_equal(uh[info_idx], u[info_idx]):
            ok += 1
    assert ok == 100, f"SC 无损校验失败: {ok}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    uh_scl, _ = scl.decode((1 - 2 * polar_encode(u)) * 20.0)
    uh_sc = sc_decode((1 - 2 * polar_encode(u)) * 20.0, frozen)
    assert np.array_equal(uh_scl, uh_sc), "L=1 SCL 应等价于 SC"

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
