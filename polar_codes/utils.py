"""工具函数：结果保存、绘图、容量计算"""
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
            w.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    "" if r.get("avg_iters") is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
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
                    "avg_iters": None
                    if row.get("avg_iters") in ("", None)
                    else float(row["avg_iters"]),
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    AWGN 连续输入容量 C = 0.5 log2(1 + 2R·Eb/N0)，用于 Shannon 参考线。
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list, dtype=np.float64)
    snr = 2.0 * rate * (10 ** (eb_n0_db_list / 10.0))
    return 0.5 * np.log2(1.0 + snr)


def find_capacity_limit(rate, eb_n0_range=(-2, 5), num_points=200):
    lo, hi = float(eb_n0_range[0]), float(eb_n0_range[1])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if compute_bpsk_capacity(mid, rate) > rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k = K if K is not None else N // 2
            info, frozen, _ = ga_construction(N, k, design_eb_n0_db)
            rate = k / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n")
            f.write(np.array2string(info, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n")
            f.write(np.array2string(frozen, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
