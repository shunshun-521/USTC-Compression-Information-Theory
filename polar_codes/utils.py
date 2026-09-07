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
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000,
                    r["avg_iters"] if r["avg_iters"] is not None else "",
                ]
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use）。
    I(X;Y) = sum_{x,y} p(x,y) log2(p(x,y)/(p(x)p(y)))
    """
    eb_n0 = 10 ** (eb_n0_db / 10.0)
    es_n0 = eb_n0 * rate
    sigma = 1.0 / np.sqrt(2.0 * es_n0)

    y = np.linspace(-8.0, 8.0, 16001)
    dy = y[1] - y[0]

    def p_y_given_x(yv, x_val):
        return np.exp(-((yv - x_val) ** 2) / (2.0 * sigma ** 2)) / (
            sigma * np.sqrt(2.0 * np.pi)
        )

    p_y = 0.5 * p_y_given_x(y, 1.0) + 0.5 * p_y_given_x(y, -1.0)
    mi = 0.0
    for x_val in (1.0, -1.0):
        p_xy = 0.5 * p_y_given_x(y, x_val)
        ratio = np.clip(p_xy / (0.5 * p_y), 1e-300, None)
        mi += np.sum(p_xy * np.log2(ratio)) * dy
    return float(mi)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""

    def capacity_minus_rate(eb_n0_db):
        return compute_bpsk_capacity(eb_n0_db, rate) - rate

    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    values = [capacity_minus_rate(x) for x in grid]
    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] <= 0:
            return brentq(capacity_minus_rate, grid[i], grid[i + 1])
    raise ValueError("Capacity limit not found in search range")


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
