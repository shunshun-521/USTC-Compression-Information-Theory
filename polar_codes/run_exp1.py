"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs("results", exist_ok=True)


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _parse_n_list(default):
    raw = os.environ.get("POLAR_N_LIST")
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ========== 参数设置 ==========
N_LIST = _parse_n_list([256, 512, 1024])
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = _env_int("POLAR_MAX_FRAMES", 100000)
MIN_ERRORS = _env_int("POLAR_MIN_ERRORS", 100)
EB_N0_RANGE = np.arange(
    _env_float("POLAR_EB_START", 0.0),
    _env_float("POLAR_EB_STOP", 5.5),
    _env_float("POLAR_EB_STEP", 0.25),
)

if __name__ == "__main__":
    import validate

    validate.main()

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}
for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, _fb=frozen_bits.copy()):
        return sc_decode(llr_ch, _fb), None

    csv_path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.environ.get("POLAR_SKIP_SIM") == "1" and os.path.isfile(csv_path):
        results = load_results_csv(csv_path)
    else:
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
        save_results_csv(results, csv_path)

    all_results[f"SC, N={N}, K={K}"] = results

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")
plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
