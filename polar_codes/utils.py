"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
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
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000,
                    "avg_iters": float(row["avg_iters"])
                    if row["avg_iters"]
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 信道互信息（bits/channel use），数值积分。
  """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        from channel import eb_n0_to_sigma

        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        grid = np.linspace(-8.0, 8.0, 20001)
        dy = grid[1] - grid[0]
        p0 = np.exp(-((grid - 1.0) ** 2) / (2 * sigma ** 2))
        p1 = np.exp(-((grid + 1.0) ** 2) / (2 * sigma ** 2))
        py = 0.5 * (p0 + p1)
        py /= np.sum(py) * dy
        p0 /= np.sum(p0) * dy
        p1 /= np.sum(p1) * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            term0 = np.where(py > 0, p0 * np.log2(p0 / py), 0.0)
            term1 = np.where(py > 0, p1 * np.log2(p1 / py), 0.0)
        mi = 0.5 * np.sum(term0 + term1) * dy
        capacities.append(float(mi))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""

    def capacity_minus_rate(eb_n0_db):
        return compute_bpsk_capacity(np.array([eb_n0_db]), rate)[0] - rate

    lo, hi = eb_n0_range
    grid = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(grid, rate)
    if np.any(caps >= rate):
        idx = np.where(caps >= rate)[0][0]
        if idx == 0:
            return float(grid[0])
        # 线性插值
        x0, x1 = grid[idx - 1], grid[idx]
        y0, y1 = caps[idx - 1], caps[idx]
        return float(x0 + (rate - y0) * (x1 - x0) / (y1 - y0))

    c_lo = capacity_minus_rate(lo)
    c_hi = capacity_minus_rate(hi)
    if c_lo * c_hi < 0:
        return float(brentq(capacity_minus_rate, lo, hi))
    return float(grid[np.argmax(caps)])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_val}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - K_val}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
