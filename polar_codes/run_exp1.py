"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import validate

validate.run_all()

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import get_simulation_params, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)

params = get_simulation_params()
N_LIST = [256, 512] if params["skip_n1024"] else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = params["max_frames"]
MIN_ERRORS = params["min_errors"]

if not np.isnan(params["eb_min"]):
    EB_N0_RANGE = np.arange(params["eb_min"], params["eb_max"] + 1e-9, params["eb_step"])
else:
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def make_decoder(fb):
        def decoder(llr_ch):
            return sc_decode(llr_ch, fb), None

        return decoder

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=make_decoder(frozen_bits),
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        verbose=True,
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

with open("results/frozen_sets.txt", "w", encoding="utf-8") as f:
    for N in N_LIST:
        K = N // 2
        info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
        f.write("=" * 53 + "\n")
        f.write(f"N={N}, K={K}, design_Eb/N0={DESIGN_EBN0} dB, R={RATE:.4f}\n")
        f.write("=" * 53 + "\n")
        f.write(f"Info indices (all {len(info_idx)}):\n")
        f.write(np.array2string(info_idx, max_line_width=120) + "\n")
        f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
        f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
        f.write("-" * 53 + "\n")

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
