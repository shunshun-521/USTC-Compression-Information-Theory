"""补跑实验三 N=512 的 BP 部分及绘图"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

n = 512
k = n // 2
rate = 0.5
design_ebn0 = 2.5
max_iter = 50
max_frames = 100000
min_errors = 100
eb_n0_range = np.arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(n, k, design_ebn0)
frozen_bits = np.ones(n, dtype=int)
frozen_bits[info_idx] = 0

bp_decoder = BPDecoder(n, frozen_bits, max_iter=max_iter)


def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters


print("N=512 BP")
r_bp = run_simulation(
    n, k, eb_n0_range, bp_d, "bp", max_frames, min_errors, verbose=True
)
save_results_csv(r_bp, f"results/exp3_bp_N{n}_R0.5.csv")

all_results = {
    "SC": load_results_csv(f"results/exp3_sc_N{n}_R0.5.csv"),
    "SCL (L=4)": load_results_csv(f"results/exp3_scl_N{n}_R0.5.csv"),
    f"BP (max_iter={max_iter})": r_bp,
}

shannon_db = find_capacity_limit(rate)
plot_bler_curves(
    all_results,
    f"SC vs SCL vs BP (N={n}, R={rate})",
    f"results/fig3_bp_N{n}_bler.png",
    shannon_limit_db=shannon_db,
)

eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
avg_iters = [r["avg_iters"] for r in r_bp]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("Avg Iterations")
ax.set_title(f"BP Average Iterations (N={n}, max_iter={max_iter})")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"results/fig3_bp_N{n}_iters.png", dpi=150)
plt.savefig(f"results/fig3_bp_N{n}_iters.pdf")
plt.close()
print("完成。")
