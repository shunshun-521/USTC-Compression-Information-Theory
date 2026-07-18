"""Experiment 3: block-length comparison at fixed rate R=0.5."""

import os

import numpy as np

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import plot_bler_curves, save_results_csv

EB_N0_LIST = list(np.arange(1.0, 5.5 + 0.25, 0.5))
BLOCK_LENGTHS = [64, 128, 256, 512]
RATE = 0.5
LIST_SIZE = 8
CRC_LENGTH = 8
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _mc_params():
    if os.environ.get("POLAR_QUICK"):
        return 500, 20
    return 100000, 100


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    max_frames, min_errors = _mc_params()

    curves = {}
    for n in BLOCK_LENGTHS:
        k = n // 2
        info_idx, _, _ = ga_construction(n, k, 2.0)
        frozen = np.ones(n, dtype=np.int8)
        frozen[info_idx] = 0
        decoder = SCLDecoder(n, frozen, list_size=LIST_SIZE, crc_length=CRC_LENGTH)

        label = f"N={n}"
        print(f"\n=== {label} (K={k}, CRC-{CRC_LENGTH}) ===")
        results = run_simulation(
            n,
            k,
            EB_N0_LIST,
            decoder.decode,
            decoder_type="scl",
            max_frames=max_frames,
            min_errors=min_errors,
            crc_length=CRC_LENGTH,
            info_indices=info_idx,
        )
        save_results_csv(results, os.path.join(RESULTS_DIR, f"exp3_N{n}.csv"))
        curves[label] = results

    plot_bler_curves(
        curves,
        f"Exp3: Block length (R={RATE}, SCL-{LIST_SIZE}+CRC)",
        os.path.join(RESULTS_DIR, "exp3_bler.png"),
    )
    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
