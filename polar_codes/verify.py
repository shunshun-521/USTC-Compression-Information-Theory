"""模块数值校验（各实验脚本可 import）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import encode_via_matrix, polar_encode


def test_encoder():
    for N in (4, 8, 16, 64):
        u = np.random.default_rng(0).integers(0, 2, size=N)
        x1 = polar_encode(u)
        x2 = encode_via_matrix(u)
        assert np.array_equal(x1, x2), f"encoder mismatch N={N}"


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)


def test_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    llr = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def run_all():
    test_encoder()
    test_sc_noiseless()
    test_scl_equals_sc()
    print("verify.py: all tests passed")


if __name__ == "__main__":
    run_all()
