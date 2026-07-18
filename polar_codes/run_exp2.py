"""Experiment 2: SCL list-size comparison."""

import os

import numpy as np

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import plot_bler_curves, save_results_csv

N, K = 128, 64
EB_N0_LIST = list(np.arange(1.0, 5.5 + 0.25, 0.5))
LIST_SIZES = [1, 2, 4, 8, 16]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _mc_params():
    if os.environ.get("POLAR_QUICK"):
        return 500, 20
    return 100000, 100


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    max_frames, min_errors = _mc_params()

    info_idx, _, _ = ga_construction(N, K, 2.0)
    frozen = np.ones(N, dtype=np.int8)
    frozen[info_idx] = 0

    curves = {}
    for list_size in LIST_SIZES:
        label = f"SCL-{list_size}"
        print(f"\n=== {label} ===")
        decoder = SCLDecoder(N, frozen, list_size=list_size)
        results = run_simulation(
            N,
            K,
            EB_N0_LIST,
            decoder.decode,
            decoder_type="scl",
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
        )
        save_results_csv(results, os.path.join(RESULTS_DIR, f"exp2_scl_L{list_size}.csv"))
        curves[label] = results

    plot_bler_curves(
        curves,
        f"Exp2: SCL list size (N={N}, K={K})",
        os.path.join(RESULTS_DIR, "exp2_bler.png"),
    )
    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
