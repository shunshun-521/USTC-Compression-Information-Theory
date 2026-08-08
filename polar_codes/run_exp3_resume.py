"""续跑实验三：补全 BP 与 N=512 仿真"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

os.makedirs("results", exist_ok=True)

# N=256: 仅 BP（SC/SCL 已完成）
N = 256
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters

print("BP 仿真 N=256 ...")
r_bp = run_simulation(
    N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
)
save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

all_results = {
    "SC": __import__("utils").load_results_csv("results/exp3_sc_N256_R0.5.csv"),
    "SCL (L=4)": __import__("utils").load_results_csv("results/exp3_scl_N256_R0.5.csv"),
    f"BP (max_iter={MAX_ITER})": r_bp,
}
shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SC vs SCL vs BP (N={N}, R={RATE})",
    f"results/fig3_bp_N{N}_bler.png",
    shannon_limit_db=shannon_db,
)
if plt:
    eb = [r["eb_n0_db"] for r in r_bp]
    iters = [r["avg_iters"] for r in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, iters, "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

# N=512 全套
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
all_results = {}

def sc_d(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None

r_sc = run_simulation(N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
all_results["SC"] = r_sc
save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

scl = SCLDecoder(N, frozen_bits, list_size=4)

def scl_d(llr_ch):
    u, _ = scl.decode(llr_ch)
    return u, None

r_scl = run_simulation(N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
all_results["SCL (L=4)"] = r_scl
save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

def bp_d512(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters

r_bp512 = run_simulation(N, K, EB_N0_RANGE, bp_d512, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
all_results[f"BP (max_iter={MAX_ITER})"] = r_bp512
save_results_csv(r_bp512, f"results/exp3_bp_N{N}_R0.5.csv")

plot_bler_curves(
    all_results,
    f"SC vs SCL vs BP (N={N}, R={RATE})",
    f"results/fig3_bp_N{N}_bler.png",
    shannon_limit_db=shannon_db,
)
if plt:
    eb = [r["eb_n0_db"] for r in r_bp512]
    iters = [r["avg_iters"] for r in r_bp512]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, iters, "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

print("实验三续跑完成。")
