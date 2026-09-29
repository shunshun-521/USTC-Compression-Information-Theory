"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import eb_n0_to_sigma, bpsk_modulate, awgn_channel, compute_llr, reorder_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_expected = np.array([1, 0, 1, 1])  # u @ B@G（Arikan 核）
    assert np.array_equal(x, x_expected), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u[info], u_hat[info]):
            errors += 1
    assert errors == 0, f"SC 高信噪比校验失败: {errors}/100 帧错误"

    u_test = np.zeros(N, dtype=int)
    u_test[info] = rng.integers(0, 2, K)
    x = polar_encode(u_test)
    llr = reorder_llr_for_decoder((1 - 2 * x) * 50.0, N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"

    if verbose:
        info8, frozen8, _ = ga_construction(8, 4, 2.5)
        print("GA N=8:", "info", info8, "frozen", frozen8)
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("GA N=256 first 20 info:", info256[:20])
        print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validation()
