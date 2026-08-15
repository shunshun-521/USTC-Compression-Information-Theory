"""Shared unit tests for polar code modules."""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """Run all verification tests. Raises AssertionError on failure."""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    uh = sc_decode(np.where(x == 0, 100.0, -100.0), np.zeros(4, dtype=bool))
    assert np.array_equal(uh, u), f"编码/译码 roundtrip 错误: {uh}"
    print("✓ 编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(15.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=15dB 下有 {errors} 帧错误"
    print("✓ SC 译码无损校验通过")

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, size=K)
    x_test = polar_encode(u_test)
    llr_test = compute_llr(bpsk_modulate(x_test), 0.01)
    u_sc = sc_decode(llr_test, frozen_bits)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("✓ SCL L=1 等价 SC 校验通过")


if __name__ == "__main__":
    run_unit_tests()
    print("\n所有单元测试通过。")
