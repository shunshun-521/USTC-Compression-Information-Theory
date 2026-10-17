"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真 CSV"""
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
                    r["avg_iters"] if r["avg_iters"] is not None else "",
                ]
            )


def load_results_csv(filepath):
    """加载 CSV 结果"""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    AWGN 信道容量（bits/channel use），Es/N0 = R * Eb/N0。
    C = 0.5 * log2(1 + 2 * Es/N0)
    """
    eb_lin = 10 ** (np.asarray(eb_n0_db_list, dtype=float) / 10.0)
    esn0 = rate * eb_lin
    return 0.5 * np.log2(1 + 2 * esn0)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    """使信道容量等于码率 R 的 Eb/N0（dB）"""
    from scipy.optimize import brentq

    def cap_minus_r(eb_db):
        return compute_bpsk_capacity([eb_db], rate)[0] - rate

    lo, hi = eb_n0_range
    try:
        return float(brentq(cap_minus_r, lo, hi))
    except ValueError:
        eb_grid = np.linspace(lo, hi, num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        idx = int(np.argmin(np.abs(caps - rate)))
        return float(eb_grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER 曲线（PNG + PDF）"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """保存各码长的信息位 / 冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db, rate=k / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={k/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N) + "\n")
            f.write("-" * 53 + "\n")
