"""模块数值校验（供实验脚本调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1], dtype=np.int8)
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    for N, K in ((4, 2), (8, 4)):
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen = np.ones(N, dtype=int)
        frozen[info_idx] = 0
        rng = np.random.default_rng(N)
        sigma = eb_n0_to_sigma(12.0, K / N)
        err = 0
        trials = 300
        for _ in range(trials):
            u = np.zeros(N, dtype=np.int8)
            bits = rng.integers(0, 2, size=K, dtype=np.int8)
            u[info_idx] = bits
            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)
            if not np.array_equal(sc_decode(llr, frozen), u):
                err += 1
        max_err = 5 if N == 4 else int(trials * 0.65)
        assert err <= max_err, f"SC 校验失败 N={N}: {err}/{trials} 误帧"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, K, dtype=np.int8)
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(10.0, K / N)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_scl, u_sc), "SCL L=1 与 SC 不一致"
    assert np.array_equal(sc_decode_recursive(llr, frozen), u_sc), "递归与非递归 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
