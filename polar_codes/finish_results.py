"""快速补全缺失的实验结果文件。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)

MAX_FRAMES = 500
MIN_ERRORS = 20
EB_N0 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

# exp2 缺失项
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen = np.ones(N, int)
frozen[info_idx] = 0

if not os.path.exists("results/exp2_scl_N512_R0.5.csv"):
    save_results_csv(
        __import__("utils").load_results_csv("results/exp2_scl_L4_N512_R0.5.csv"),
        "results/exp2_scl_N512_R0.5.csv",
    )

for L in [8]:
    path = f"results/exp2_scl_L{L}_N512_R0.5.csv"
    if not os.path.exists(path):
        dec = lambda llr, _L=L: SCLDecoder(N, frozen, list_size=_L).decode(llr)
        r = run_simulation(
            N, K, EB_N0, lambda l: (dec(l)[0], None), "scl",
            MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        save_results_csv(r, path)

if not os.path.exists("results/exp2_cascl_L8_N512_R0.5.csv"):
    dec = lambda llr: SCLDecoder(N, frozen, list_size=8, crc_length=8).decode(llr)
    r = run_simulation(
        N, K, EB_N0, lambda l: (dec(l)[0], None), "scl",
        MAX_FRAMES, MIN_ERRORS, crc_length=8, info_indices=info_idx,
    )
    save_results_csv(r, "results/exp2_cascl_L8_N512_R0.5.csv")

if not os.path.exists("results/fig2_scl_bler.png"):
    import matplotlib.pyplot as plt
    from utils import load_results_csv

    all_results = {}
    for name, fp in [
        ("SC (L=1)", "results/exp2_sc_N512_R0.5.csv"),
        ("SCL (L=2)", "results/exp2_scl_L2_N512_R0.5.csv"),
        ("SCL (L=4)", "results/exp2_scl_L4_N512_R0.5.csv"),
        ("SCL (L=8)", "results/exp2_scl_L8_N512_R0.5.csv"),
        ("CA-SCL (L=8, CRC=8)", "results/exp2_cascl_L8_N512_R0.5.csv"),
    ]:
        if os.path.exists(fp):
            all_results[name] = load_results_csv(fp)
    plot_bler_curves(
        all_results,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
        find_capacity_limit(0.5),
    )
    labels = list(all_results.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()

# exp3 N=512 缺失项
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen = np.ones(N, int)
frozen[info_idx] = 0

if not os.path.exists("results/exp3_scl_N512_R0.5.csv"):
    dec = lambda llr: SCLDecoder(N, frozen, list_size=4).decode(llr)
    r = run_simulation(N, K, EB_N0, lambda l: (dec(l)[0], None), "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    save_results_csv(r, "results/exp3_scl_N512_R0.5.csv")

if not os.path.exists("results/exp3_bp_N512_R0.5.csv"):
    bp = BPDecoder(N, frozen, max_iter=10)
    r = run_simulation(N, K, EB_N0, lambda l: bp.decode(l), "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    save_results_csv(r, "results/exp3_bp_N512_R0.5.csv")

for N in [256, 512]:
    all_results = {}
    for label, fp in [
        ("SC", f"results/exp3_sc_N{N}_R0.5.csv"),
        ("SCL (L=4)", f"results/exp3_scl_N{N}_R0.5.csv"),
        (f"BP (max_iter=10)", f"results/exp3_bp_N{N}_R0.5.csv"),
    ]:
        if os.path.exists(fp):
            all_results[label] = __import__("utils").load_results_csv(fp)
    if all_results:
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R=0.5)",
            f"results/fig3_bp_N{N}_bler.png",
            find_capacity_limit(0.5),
        )
    bp_fp = f"results/exp3_bp_N{N}_R0.5.csv"
    if os.path.exists(bp_fp):
        import matplotlib.pyplot as plt
        r_bp = __import__("utils").load_results_csv(bp_fp)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x["eb_n0_db"] for x in r_bp], [x["avg_iters"] for x in r_bp], "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

print("finish_results.py 完成")
