"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate
from simulation import fast_sim_settings, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_validate():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)

    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 100.0 * bpsk_modulate(polar_encode(u))
        assert np.array_equal(sc_decode(llr, fb), u)

    scl_path = os.path.join(os.path.dirname(__file__), "decoder_scl.py")
    with open(scl_path) as f:
        assert "SCLDecoder" in f.read()
    print("单元测试通过。")


if __name__ == "__main__":
    if os.environ.get("POLAR_RUN_VALIDATE", "1") == "1":
        run_validate()

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    sim = fast_sim_settings()
    MAX_FRAMES = sim["max_frames"]
    MIN_ERRORS = sim["min_errors"]
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, os.path.join(results_dir, "frozen_sets.txt"))

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        fb = frozen_bits.astype(bool)

        def decoder(llr_ch):
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
        )

        label = f"SC, N={N}, K={K}"
        all_results[label] = results
        save_results_csv(results, os.path.join(results_dir, f"exp1_sc_N{N}_R0.5.csv"))

    shannon_db = find_capacity_limit(RATE)
    print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path=os.path.join(results_dir, "fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")
