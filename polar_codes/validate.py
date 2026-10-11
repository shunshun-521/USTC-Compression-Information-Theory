"""模块数值校验（实验脚本导入前调用）"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_validations(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u, apply_bit_reversal=False)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    G %= 2
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G 不一致: {x} vs {x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    sc_ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        codeword = polar_encode(u, apply_bit_reversal=False)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(codeword), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        sc_ok += int(np.array_equal(u_hat[info_idx], u[info_idx]))
    assert sc_ok == 100, f"SC 校验失败: {sc_ok}/100"

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    cw = polar_encode(u_test, apply_bit_reversal=False)
    llr = compute_llr(bpsk_modulate(cw), eb_n0_to_sigma(12, 0.5))
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"

    if verbose:
        print("validate.py: 全部校验通过")
    return True


if __name__ == "__main__":
    run_validations()
