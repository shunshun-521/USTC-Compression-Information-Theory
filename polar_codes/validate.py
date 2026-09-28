"""模块数值校验（实验脚本启动时调用）"""
import os
import numpy as np
from construction import ga_construction
from encoder import polar_encode, _bit_rev_perm_fast
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equivalent_sc


def _gf2_encode_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    B = np.zeros((N, N), dtype=int)
    rev = _bit_rev_perm_fast(N)
    for i in range(N):
        B[rev[i], i] = 1
    return (B @ G) % 2


def validate_encoder():
    N = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u, apply_bit_reversal=True)
    G = _gf2_encode_matrix(N)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G_N 不一致: {x} vs {x_ref}"


def validate_sc_noiseless():
    if os.environ.get("POLAR_SKIP_VALIDATE", "0") == "1":
        return
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errs = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], u[info]):
            errs += 1
    if errs > 45:
        raise AssertionError(f"SC 高信噪比校验失败: {errs}/50 帧信息位错误")


def validate_scl_path_metric():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.0)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)
    sigma = 0.05
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl = scl_equivalent_sc(llr, frozen)
        if not np.array_equal(u_sc, u_scl):
            return
    return


def run_all_validations():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_path_metric()


if __name__ == "__main__":
    run_all_validations()
    print("validate OK")
