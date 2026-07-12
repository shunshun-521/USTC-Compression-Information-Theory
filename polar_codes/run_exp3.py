"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 与 SC、SCL（L=4）对比
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), (u @ build_generator_matrix(4)) % 2)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(11)
    for _ in range(50):
        u64 = np.zeros(N, dtype=int)
        u64[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u64)), sigma, rng), sigma
        )
        assert np.array_equal(sc_decode(llr, fb), sc_decode(llr, fb))
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 50000
MAX_FRAMES_SCL = 1500
MAX_FRAMES_BP = 30000
MIN_ERRORS = 100
MIN_ERRORS_SCL = 50
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

if __name__ == "__main__":
    run_unit_tests()

    for N in N_LIST:
        K = N // 2
        info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\n实验三 SC: N={N}")
        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr_ch):
            u, pm = SCLDecoder(
                N, frozen_bits.astype(bool), list_size=4
            ).decode(llr_ch)
            return u, None

        print(f"SCL L=4: N={N}")
        scl_csv = f"results/exp3_scl_N{N}_R0.5.csv"
        if os.path.exists(scl_csv):
            from utils import load_results_csv
            r_scl = load_results_csv(scl_csv)
            print(f"  已加载 {scl_csv}")
        else:
            r_scl = run_simulation(
                N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES_SCL, MIN_ERRORS_SCL,
                info_indices=info_idx,
            )
            save_results_csv(r_scl, scl_csv)

        all_results["SCL (L=4)"] = r_scl

        bp_csv = f"results/exp3_bp_N{N}_R0.5.csv"
        bp_decoder = BPDecoder(N, frozen_bits.astype(bool), max_iter=MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"BP: N={N}")
        if os.path.exists(bp_csv):
            from utils import load_results_csv
            r_bp = load_results_csv(bp_csv)
            print(f"  已加载 {bp_csv}")
        else:
            r_bp = run_simulation(
                N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES_BP, MIN_ERRORS,
                info_indices=info_idx,
            )
            save_results_csv(r_bp, bp_csv)
        all_results[f"BP (max_iter={MAX_ITER})"] = r_bp

        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("\n实验三完成。")
