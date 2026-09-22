"""极化码模块数值正确性校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests(verbose=True):
    """运行所有单元测试，失败时抛出 AssertionError。"""
    if verbose:
        print("=" * 50)
        print("极化码单元测试")
        print("=" * 50)

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    if verbose:
        print("[PASS] 编码器校验")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print(f"N=8, K=4: info={info8}, frozen={frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print(f"N=256, K=128, info前20: {info256[:20]}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u_sent[info_idx]), "SC 译码失败"
    if verbose:
        print("[PASS] SC 译码无损校验 (N=64, 100帧, Eb/N0=10dB)")

    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"
    if verbose:
        print("[PASS] SCL(L=1) 等价于 SC")

    if verbose:
        print("所有单元测试通过。\n")


if __name__ == "__main__":
    run_unit_tests()
