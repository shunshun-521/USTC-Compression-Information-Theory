"""极化码模块快速验证脚本"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def main():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8 info:", info8, "frozen:", frozen8)

    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), [1, 1, 0, 1])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    for _ in range(50):
        u0 = np.zeros(N, dtype=np.int8)
        u0[info] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u0)), 0.1)
        assert np.array_equal(u0, sc_decode(llr, frozen))
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u0, u_scl)

    print("validate.py: all checks passed")


if __name__ == "__main__":
    main()
