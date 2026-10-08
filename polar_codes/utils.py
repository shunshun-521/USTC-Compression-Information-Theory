"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    cols = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in cols}
            w.writerow(row)


def load_results_csv(filepath):
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def compute_bpsk_capacity(eb_n0_db_list, rate):
    snr_list = 2.0 * rate * (10.0 ** (np.asarray(eb_n0_db_list, dtype=float) / 10.0))

    def integrand(y, s):
        return np.log2(1.0 + np.exp(-2.0 * s * y)) * np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)

    caps = []
    for s in snr_list:
        val, _ = integrate.quad(lambda y: integrand(y, s), -np.inf, np.inf)
        caps.append(1.0 - val)
    return np.asarray(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=2000):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    if np.all(caps > rate):
        return float(grid[0])
    if np.all(caps < rate):
        return float(grid[-1])
    idx_arr = np.where(caps >= rate)[0]
    if len(idx_arr) == 0:
        return float(grid[-1])
    idx = idx_arr[0]
    if idx == 0:
        return float(grid[0])
    x0, x1 = grid[idx - 1], grid[idx]
    c0, c1 = caps[idx - 1], caps[idx]
    return float(x0 + (rate - c0) * (x1 - x0) / (c1 - c0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        plt.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon (R) ~ {shannon_limit_db:.2f} dB")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    if K is None:
        pass
    with open(save_path, "w") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            rate = K_use / N
            info, frozen, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n{info}\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n{frozen}\n")
            f.write("-" * 53 + "\n")
