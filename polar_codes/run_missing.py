"""补全缺失的实验结果（exp2 SCL L=8, exp3 BP N=512）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)

N = 512
K = N // 2
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)
MAX_ITER = 50

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

# exp2 SCL L=8
print("SCL L=8 仿真...")
scl_path = f"results/exp2_scl_L8_N{N}_R0.5.csv"
if not os.path.exists(scl_path):

    def scl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=0).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
    )
    save_results_csv(results, scl_path)
    print(f"已保存 {scl_path}")
else:
    print(f"已存在 {scl_path}，跳过")

# exp3 BP N=512
print("BP N=512 仿真...")
bp_path = f"results/exp3_bp_N{N}_R0.5.csv"
if not os.path.exists(bp_path):
    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        u_hat, num_iters = bp_decoder.decode(llr_ch)
        return u_hat, num_iters

    r_bp = run_simulation(
        N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
    )
    save_results_csv(r_bp, bp_path)

    # 加载已有 SC/SCL 结果绘图
    from utils import load_results_csv

    all_results = {
        "SC": load_results_csv(f"results/exp3_sc_N{N}_R0.5.csv"),
        "SCL (L=4)": load_results_csv(f"results/exp3_scl_N{N}_R0.5.csv"),
        f"BP (max_iter={MAX_ITER})": r_bp,
    }
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
    print(f"已保存 {bp_path} 及 fig3_bp_N{N}_*")
else:
    print(f"已存在 {bp_path}，跳过")

print("补全实验完成。")
