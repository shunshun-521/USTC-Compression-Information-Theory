"""模块数值正确性校验（各实验脚本开头调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_no_br
from channel import bpsk_modulate, awgn_channel, eb_n0_to_sigma, channel_llr_for_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def _gen_matrix_encode(u):
    """u @ (B F^{⊗n}) 用于核对编码器。"""
    N = len(u)
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    G %= 2
    n = int(np.log2(N))
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        v, r = i, 0
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        B[r, i] = 1
    return (u @ (B @ G)) % 2


def run_validations():
    """运行全部单元测试，失败则抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = _gen_matrix_encode(u)
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    fb = frozen.astype(bool)
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_fails = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = channel_llr_for_decoder(y, sigma, N)
        u_hat = sc_decode(llr, fb)
        if not np.array_equal(u_hat, u):
            sc_fails += 1
    assert sc_fails == 0, f"SC 高信噪比校验失败: {sc_fails}/100 帧有错"

    u_test = np.zeros(N, dtype=int)
    u_test[info] = rng.integers(0, 2, K)
    x = polar_encode(u_test)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = channel_llr_for_decoder(y, sigma, N)
    u_sc = sc_decode(llr, fb)
    u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"

    print("[validations] 全部校验通过。")


if __name__ == "__main__":
    run_validations()
