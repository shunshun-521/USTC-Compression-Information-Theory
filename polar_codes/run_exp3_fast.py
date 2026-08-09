"""快速完成实验三剩余仿真（BP 使用 max_frames=50000 以在合理时间内完成）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)
BP_MAX_FRAMES = 50000
SC_MAX_FRAMES = 100000

os.makedirs("results", exist_ok=True)


def run_bp(N, info_idx, frozen_bits):
    bp = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        u_hat, num_iters = bp.decode(llr_ch)
        return u_hat, num_iters

    return run_simulation(
        N,
        N // 2,
        EB_N0_RANGE,
        bp_d,
        "bp",
        BP_MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info_idx,
        verbose=True,
    )


# N=256 BP
N = 256
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

print("BP N=256 ...")
r_bp256 = run_bp(N, info_idx, frozen_bits)
save_results_csv(r_bp256, "results/exp3_bp_N256_R0.5.csv")

all256 = {
    "SC": load_results_csv("results/exp3_sc_N256_R0.5.csv"),
    "SCL (L=4)": load_results_csv("results/exp3_scl_N256_R0.5.csv"),
    f"BP (max_iter={MAX_ITER})": r_bp256,
}
shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all256,
    f"SC vs SCL vs BP (N={N}, R={RATE})",
    "results/fig3_bp_N256_bler.png",
    shannon_limit_db=shannon_db,
)
if plt:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["eb_n0_db"] for r in r_bp256], [r["avg_iters"] for r in r_bp256], "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/fig3_bp_N256_iters.png", dpi=150)
    plt.savefig("results/fig3_bp_N256_iters.pdf")
    plt.close()

# N=512 全套
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
all512 = {}


def sc_d(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None


print("SC N=512 ...")
r_sc = run_simulation(
    N, K, EB_N0_RANGE, sc_d, "sc", SC_MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
)
all512["SC"] = r_sc
save_results_csv(r_sc, "results/exp3_sc_N512_R0.5.csv")

scl = SCLDecoder(N, frozen_bits, list_size=4)

def scl_d(llr_ch):
    u, _ = scl.decode(llr_ch)
    return u, None


print("SCL N=512 ...")
r_scl = run_simulation(
    N, K, EB_N0_RANGE, scl_d, "scl", SC_MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
)
all512["SCL (L=4)"] = r_scl
save_results_csv(r_scl, "results/exp3_scl_N512_R0.5.csv")

print("BP N=512 ...")
r_bp512 = run_bp(N, info_idx, frozen_bits)
all512[f"BP (max_iter={MAX_ITER})"] = r_bp512
save_results_csv(r_bp512, "results/exp3_bp_N512_R0.5.csv")

plot_bler_curves(
    all512,
    f"SC vs SCL vs BP (N={N}, R={RATE})",
    "results/fig3_bp_N512_bler.png",
    shannon_limit_db=shannon_db,
)
if plt:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["eb_n0_db"] for r in r_bp512], [r["avg_iters"] for r in r_bp512], "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/fig3_bp_N512_iters.png", dpi=150)
    plt.savefig("results/fig3_bp_N512_iters.pdf")
    plt.close()

# 规范文件名：exp2_scl_N512_R0.5.csv（L=4 代表）
import shutil
shutil.copy(
    "results/exp2_scl_L4_N512_R0.5.csv",
    "results/exp2_scl_N512_R0.5.csv",
)

print("实验三快速完成。")
