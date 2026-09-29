"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validate import run_all as run_validate
from construction import ga_construction, info_set_for_codec
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), "results")

if not os.environ.get("POLAR_SKIP_VALIDATE"):
    run_validate()

N_LIST = [int(x) for x in os.environ.get("POLAR_N_LIST", "256,512,1024").split(",")]
if os.environ.get("POLAR_SKIP_N1024"):
    N_LIST = [n for n in N_LIST if n != 1024]
RATE = 0.5
DESIGN_EBN0 = float(os.environ.get("POLAR_DESIGN_EBN0", "2.5"))
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, os.path.join(RESULTS, "frozen_sets.txt"))

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx = info_set_for_codec(N, K)
    _, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        frozen_bits=frozen_bits,
        info_indices=info_idx,
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, os.path.join(RESULTS, f"exp1_sc_N{N}_R0.5.csv"))

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
    shannon_limit_db=shannon_db,
)
print(f"\n实验一完成。结果保存至 {RESULTS}/")
