"""
生成规格化仿真结果（N=256/512，中等帧数，用于报告核对）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv

os.makedirs("results", exist_ok=True)

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 5000
MIN_ERRORS = 30
EB_N0_RANGE = np.arange(1.0, 5.0, 0.5)

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print(f"Generating exp1_sc_N{N}_R0.5.csv ...")
    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits,
        verbose=True,
    )
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

print("Done.")
