"""Experiment 1: SC vs SCL vs BP decoder comparison."""

import os

import numpy as np

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

N, K = 128, 64
EB_N0_LIST = list(np.arange(0.0, 5.5 + 0.25, 0.5))
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

    scl_decoder = SCLDecoder(N, frozen, list_size=8)
    bp_decoder = BPDecoder(N, frozen, max_iter=50)

    decoders = {
        "SC": (lambda llr: (sc_decode(llr, frozen), None), "sc"),
        "SCL-8": (lambda llr: scl_decoder.decode(llr), "scl"),
        "BP": (lambda llr: bp_decoder.decode(llr), "bp"),
    }

    curves = {}
    for name, (decoder_fn, decoder_type) in decoders.items():
        print(f"\n=== {name} ===")
        results = run_simulation(
            N,
            K,
            EB_N0_LIST,
            decoder_fn,
            decoder_type=decoder_type,
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
        )
        save_results_csv(results, os.path.join(RESULTS_DIR, f"exp1_{name.lower().replace('-', '_')}.csv"))
        curves[name] = results

    shannon = find_capacity_limit(K / N)
    plot_bler_curves(
        curves,
        f"Exp1: SC vs SCL vs BP (N={N}, K={K})",
        os.path.join(RESULTS_DIR, "exp1_bler.png"),
        shannon_limit_db=shannon,
    )
    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
