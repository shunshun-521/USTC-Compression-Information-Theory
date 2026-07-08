"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            writer.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
            ])


def load_results_csv(filepath):
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
            })
    return results


def _log2_1pexp(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = x[pos] / np.log(2.0) + np.log1p(np.exp(-x[pos])) / np.log(2.0)
    out[~pos] = np.log1p(np.exp(x[~pos])) / np.log(2.0)
    return out


def _bpsk_capacity_scalar(eb_n0_db, rate):
    snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))

    def integrand(y):
        x = -2.0 * snr - 2.0 * np.sqrt(snr) * y
        return _log2_1pexp(x) * np.exp(-y * y / 2.0) / np.sqrt(2.0 * np.pi)

    val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
    return max(0.0, 1.0 - val)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    return np.array([_bpsk_capacity_scalar(eb, rate) for eb in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=200):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)

    if np.any(caps >= rate):
        idx = np.where(caps >= rate)[0][0]
        if idx == 0:
            return float(grid[0])
        e0, e1 = grid[idx - 1], grid[idx]
        c0, c1 = caps[idx - 1], caps[idx]
        return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))

    def func(eb):
        return _bpsk_capacity_scalar(eb, rate) - rate

    try:
        return float(brentq(func, eb_n0_range[0], eb_n0_range[1]))
    except ValueError:
        return float(eb_n0_range[1])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, linewidth=2, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                k = N // 2
            elif isinstance(K, dict):
                k = K[N]
            else:
                k = K
            rate = k / N
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
