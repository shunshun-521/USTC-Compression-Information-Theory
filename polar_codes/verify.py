"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # G_N = B_N F^{⊗n}，x = u G_N
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info, "frozen:", frozen)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errs = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errs += 1
    assert errs == 0, f"SC 无损测试失败: {errs} 帧错误"

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat, u_hat_r), "递归与非递归 SC 不一致"

    u_hat_sc = sc_decode(llr, frozen_bits)
    u_hat_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_hat_sc, u_hat_scl), "SCL L=1 与 SC 不一致"

    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
