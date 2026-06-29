"""补充运行缺失的仿真结果（N=512/1024 for exp1, N=512 for exp3）。"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs("results", exist_ok=True)

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))

# ---------- Exp1: N=512, 1024 ----------
EB_N0_EXP1 = np.arange(0.0, 5.5, 0.25)
all_exp1 = {}

if os.path.exists("results/exp1_sc_N256_R0.5.csv"):
    all_exp1["SC, N=256, K=128"] = load_results_csv("results/exp1_sc_N256_R0.5.csv")

for N in [512, 1024]:
    K = N // 2
    print(f"\nExp1 SC: N={N}, K={K}")
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bool), None

    results = run_simulation(
        N=N, K=K, eb_n0_db_list=EB_N0_EXP1, decoder=decoder,
        decoder_type="sc", max_frames=MAX_FRAMES, min_errors=MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_exp1[f"SC, N={N}, K={K}"] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_exp1,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("Exp1 补充完成。")

# ---------- Exp3: N=512 ----------
EB_N0_EXP3 = np.arange(1.0, 5.5, 0.25)
MAX_ITER = 50
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
frozen_bool = frozen_bits.astype(bool)
all_exp3 = {}

def sc_d(llr_ch):
    return sc_decode(llr_ch, frozen_bool), None

print(f"\nExp3 N={N} SC")
r_sc = run_simulation(N, K, EB_N0_EXP3, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
                      info_indices=info_idx, verbose=True)
all_exp3["SC"] = r_sc
save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

def scl_d(llr_ch):
    u, pm = SCLDecoder(N, frozen_bool, list_size=4).decode(llr_ch)
    return u, None

print(f"\nExp3 N={N} SCL")
r_scl = run_simulation(N, K, EB_N0_EXP3, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
                       info_indices=info_idx, verbose=True)
all_exp3["SCL (L=4)"] = r_scl
save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

bp_decoder = BPDecoder(N, frozen_bool, max_iter=MAX_ITER)

def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters

print(f"\nExp3 N={N} BP")
r_bp = run_simulation(N, K, EB_N0_EXP3, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
                      info_indices=info_idx, verbose=True)
all_exp3[f"BP (max_iter={MAX_ITER})"] = r_bp
save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

plot_bler_curves(
    all_exp3,
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
print("Exp3 N=512 补充完成。")
