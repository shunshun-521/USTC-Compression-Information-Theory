"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
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
        return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use），Es/N0 = Eb/N0 * R。"""
    caps = []
    for eb_n0_db in eb_n0_db_list:
        es_n0 = rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(t):
            return np.log(1.0 + np.exp(-es_n0 * t * t)) * np.exp(-0.5 * t * t)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        val /= np.sqrt(2.0 * np.pi) * np.log(2)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        # 二分细化
        lo, hi = eb_n0_range[0], eb_n0_range[1]
        for _ in range(50):
            mid = (lo + hi) / 2.0
            c = compute_bpsk_capacity([mid], rate)[0]
            if c > rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    return float(eb_grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    if K is None:
        K = "N//2"
    lines = []
    for N in N_list:
        K_val = N // 2 if K == "N//2" or K is None else K
        info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
        lines.append("=" * 53)
        lines.append(
            f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={K_val / N:.4f}"
        )
        lines.append("=" * 53)
        lines.append(f"Info indices (all {len(info_idx)}):")
        lines.append(np.array2string(info_idx, threshold=N))
        lines.append(f"Frozen indices (all {len(frozen_idx)}):")
        lines.append(np.array2string(frozen_idx, threshold=N))
        lines.append("-" * 53)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
