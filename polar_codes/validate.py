"""单元测试：编码器与各译码器基础正确性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u))
        llr = compute_llr(y, 1e-8)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])


def validate_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(1)
    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N))), 0.1)
    u_sc, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_l1_equals_sc()
    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()
