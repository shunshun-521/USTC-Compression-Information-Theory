"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from verify import main as run_verify

run_verify()

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), "results")

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
SKIP_N1024 = os.environ.get("POLAR_SKIP_N1024", "0") == "1"
eb_min = float(os.environ.get("POLAR_EB_MIN", "0.0"))
eb_max = float(os.environ.get("POLAR_EB_MAX", "5.25"))
eb_step = float(os.environ.get("POLAR_EB_STEP", "0.25"))
EB_N0_RANGE = np.arange(eb_min, eb_max + 1e-9, eb_step)

if SKIP_N1024:
    N_LIST = [n for n in N_LIST if n != 1024]

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, os.path.join(RESULTS, "frozen_sets.txt"))

all_results = {}
for N in N_LIST:
    K = N // 2
    print(f"\n{'='*60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'='*60}")
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits.astype(bool)), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
    )
    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, os.path.join(RESULTS, f"exp1_sc_N{N}_R0.5.csv"))

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。")
