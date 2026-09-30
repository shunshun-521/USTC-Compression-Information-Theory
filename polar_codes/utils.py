"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
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
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            out.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters") and row["avg_iters"]
                    else None,
                }
            )
        return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    eb = np.asarray(eb_n0_db_list, dtype=np.float64)
    caps = []
    x = np.linspace(-12.0, 12.0, 120001)
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
    for eb_db in eb:
        es_n0 = (10.0 ** (eb_db / 10.0)) * rate
        integrand = np.log2(1.0 + np.exp(-2.0 * es_n0 * x ** 2))
        caps.append(1.0 - np.trapezoid(integrand * pdf, x))
    return np.asarray(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(grid) - 1:
        return float(grid[idx])
    e0, e1 = grid[idx - 1], grid[idx]
    c0, c1 = caps[idx - 1], caps[idx]
    if c1 == c0:
        return float(e1)
    t = (rate - c0) / (c1 - c0)
    return float(e0 + t * (e1 - e0))


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            rate = K_use / N
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=10000) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=10000) + "\n")
            f.write("-" * 53 + "\n")
