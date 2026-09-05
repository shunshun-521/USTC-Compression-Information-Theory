"""仅补跑实验二 CA-SCL 部分并更新 fig2。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv

N = 512
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

if __name__ == "__main__":
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def cascl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
        return u_hat, None

    print(f"CA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")
    results = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, "scl",
        MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
        info_indices=info_idx, frozen_bits=frozen_bits,
    )
    save_results_csv(results, f"results/exp2_cascl_L8_N{N}_R0.5.csv")
    print("CA-SCL 完成。")
