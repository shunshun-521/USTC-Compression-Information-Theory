"""单元测试与模块验证"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, verify_sc_decoders
from decoder_scl import SCLDecoder, verify_scl_equals_sc
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
    u2 = np.zeros(4, dtype=int)
    u2[:] = u
    x2 = polar_encode(u2)
    assert np.array_equal(x, x2)


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6])
    assert np.array_equal(frozen, [1, 2, 4, 7])


def test_sc():
    verify_sc_decoders(64, num_trials=100, eb_n0_db=10.0)
    assert np.array_equal(
        sc_decode_recursive(np.array([100.0, -100.0]), np.array([False, False])),
        sc_decode(np.array([100.0, -100.0]), np.array([False, False])),
    )


def test_scl():
    verify_scl_equals_sc(64, num_trials=20)


def main():
    print("Running polar code validation...")
    test_encoder()
    print("  encoder OK")
    test_construction()
    print("  construction OK")
    test_sc()
    print("  SC decoder OK")
    test_scl()
    print("  SCL decoder OK")
    print("All validation tests passed.")


if __name__ == "__main__":
    main()
