"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 与 SC、SCL（L=4）对比
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from channel import awgn_channel, bpsk_modulate, compute_llr
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
assert np.array_equal(polar_encode(u), [1, 1, 0, 1])

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=bool)
frozen_t[info_t] = False
rng = np.random.default_rng(0)
for _ in range(100):
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = rng.integers(0, 2, size=K_t)
    llr_t = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), 0.001, rng), 0.001)
    assert np.array_equal(sc_decode(llr_t, frozen_t), u_t)

for _ in range(20):
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = rng.integers(0, 2, size=K_t)
    llr_t = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), 0.001, rng), 0.001)
    u_scl, _ = SCLDecoder(N_t, frozen_t, list_size=1).decode(llr_t)
    assert np.array_equal(sc_decode(llr_t, frozen_t), u_scl)

N_bp, K_bp = 32, 16
info_bp, _, _ = ga_construction(N_bp, K_bp, 2.5)
frozen_bp = np.ones(N_bp, dtype=bool)
frozen_bp[info_bp] = False
u_bp = np.zeros(N_bp, dtype=int)
u_bp[info_bp] = rng.integers(0, 2, size=K_bp)
llr_bp = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_bp)), 0.001, rng), 0.001)
u_hat_bp, _ = BPDecoder(N_bp, frozen_bp).decode(llr_bp)
assert np.array_equal(u_hat_bp, u_bp)
print("单元测试通过。\n")

os.makedirs("results", exist_ok=True)

N_LIST = [int(x) for x in os.environ.get("POLAR_N_LIST", "256,512").split(",")]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_MIN = float(os.environ.get("POLAR_EB_MIN", "1.0"))
EB_N0_MAX = float(os.environ.get("POLAR_EB_MAX", "5.25"))
EB_N0_STEP = float(os.environ.get("POLAR_EB_STEP", "0.25"))
EB_N0_RANGE = np.arange(EB_N0_MIN, EB_N0_MAX, EB_N0_STEP)

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    all_results = {}

    def sc_d(llr_ch, fb=frozen_bits):
        return sc_decode(llr_ch, fb), None

    print(f"\n{'=' * 60}\nN={N}, SC 仿真\n{'=' * 60}")
    r_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx,
    )
    all_results["SC"] = r_sc
    save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

    def scl_d(llr_ch, fb=frozen_bits):
        u, _ = SCLDecoder(N, fb, list_size=4).decode(llr_ch)
        return u, None

    print(f"\nN={N}, SCL L=4 仿真")
    r_scl = run_simulation(
        N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx,
    )
    all_results["SCL (L=4)"] = r_scl
    save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch, bp=bp_decoder):
        u_hat, num_iters = bp.decode(llr_ch)
        return u_hat, num_iters

    print(f"\nN={N}, BP 仿真")
    r_bp = run_simulation(
        N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx,
    )
    all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
    save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

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
