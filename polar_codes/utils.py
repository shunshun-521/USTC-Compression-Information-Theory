"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "eb_n0_db",
                "bler",
                "ber",
                "num_errors",
                "num_frames",
                "avg_decode_time_ms",
                "avg_iters",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    r["avg_iters"] if r["avg_iters"] is not None else "",
                ]
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db, rate):
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

    def integrand(y):
        return np.log2(1.0 + np.exp(-2.0 * snr * y)) * np.exp(-y * y) / np.sqrt(np.pi)

    val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    return 1.0 - val


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = [compute_bpsk_capacity(eb, rate) for eb in eb_grid]
    for i in range(len(eb_grid) - 1):
        if caps[i] <= rate <= caps[i + 1]:
            return float(np.interp(rate, [caps[i], caps[i + 1]], [eb_grid[i], eb_grid[i + 1]]))
    return float(eb_grid[-1])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, marker="o", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_use = N // 2
            else:
                K_use = K
            info, frozen, _ = ga_construction(N, K_use, design_eb_n0_db)
            rate = K_use / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_use}):\n{info}\n")
            f.write(f"Frozen indices (all {N - K_use}):\n{frozen}\n")
            f.write("-" * 53 + "\n")
