"""极化码编译码仿真验证脚本"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from simulation import run_unit_tests


def main():
    run_unit_tests()

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8.tolist())
    print("frozen_indices:", frozen8.tolist())

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20].tolist())

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"Encoder: u={u.tolist()} -> x={x.tolist()}")


if __name__ == "__main__":
    main()
