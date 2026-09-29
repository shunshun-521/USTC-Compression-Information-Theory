"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, decode_sc_from_channel_llr
from decoder_scl import SCLDecoder
from encoder import bit_reversal_permutation


def run_validate(verbose=True):
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    expected = (G @ u) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = decode_sc_from_channel_llr(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码失败"

    # SCL L=1 等价 SC
    br = bit_reversal_permutation(N)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    llr_in = np.zeros(N)
    llr_in[br] = llr
    u_sc = sc_decode(llr_in, frozen.astype(bool))
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_in)
    assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"

    if verbose:
        print("validate: 全部通过")
    return True


if __name__ == "__main__":
    run_validate()
