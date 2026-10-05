"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = dict(r)
            row["avg_decode_time"] = row["avg_decode_time"] * 1000.0
            if row["avg_iters"] is None:
                row["avg_iters"] = ""
            w.writerow(row)


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"]) / 1000.0
            row["avg_iters"] = float(row["avg_iters"]) if row["avg_iters"] != "" else None
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    from scipy import integrate

    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        def integrand(y):
            t = -2.0 * snr * y
            t = np.clip(t, -700.0, 700.0)
            return np.log2(1.0 + np.exp(t)) * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -50.0, 50.0, limit=200)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    return float(grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, linewidth=1.5, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon ≈ {shannon_limit_db:.2f} dB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            rate = K_n / N
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_n}):\n{np.array2string(info_idx, threshold=K_n)}\n")
            f.write(f"Frozen indices (all {N-K_n}):\n{np.array2string(frozen_idx, threshold=N-K_n)}\n")
            f.write("-" * 53 + "\n")
