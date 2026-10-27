"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_fail = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, fb), u):
            sc_fail += 1
    assert sc_fail == 0, f"SC 无损校验失败: {sc_fail}/100"

    llr = np.array([10.0, -10.0, 5.0, -5.0] + [8.0] * (N - 4))
    u1 = sc_decode(llr, fb)
    u2, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(u1, u2), "L=1 SCL 应与 SC 等价"

    u3, _ = BPDecoder(N, fb, max_iter=40).decode(llr)
    assert u3.shape == (N,)

    print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_all_tests()
