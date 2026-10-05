"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from env_config import eb_n0_range, parse_n_list, sim_config, skip_n1024
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv
from verify import run_unit_tests

os.makedirs("results", exist_ok=True)

run_unit_tests()

N_LIST = skip_n1024(parse_n_list([256, 512, 1024]))
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES, MIN_ERRORS, _, _, _ = sim_config()
EB_N0_RANGE = eb_n0_range(np.arange(0.0, 5.5, 0.25))

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = int(N * RATE)
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, fb=frozen_bits.astype(bool)):
        return sc_decode(llr_ch, fb), None

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
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
