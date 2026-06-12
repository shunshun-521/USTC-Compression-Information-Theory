"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
        for row in results:
            writer.writerow(
                {
                    "eb_n0_db": row["eb_n0_db"],
                    "bler": row["bler"],
                    "ber": row["ber"],
                    "num_errors": row["num_errors"],
                    "num_frames": row["num_frames"],
                    "avg_decode_time_ms": row["avg_decode_time"] * 1000.0,
                    "avg_iters": "" if row["avg_iters"] is None else row["avg_iters"],
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": None
                    if row["avg_iters"] in ("", None)
                    else float(row["avg_iters"]),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    与 channel.eb_n0_to_sigma 的噪声模型一致。
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        sigma = (2.0 * rate * (10.0 ** (eb_n0_db / 10.0))) ** -0.5
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

        def integrand(noise):
            llr = 2.0 * (1.0 + noise) / (sigma ** 2)
            p0 = 1.0 / (1.0 + np.exp(-llr))
            h = -p0 * np.log2(p0 + 1e-300) - (1.0 - p0) * np.log2(1.0 - p0 + 1e-300)
            return h * norm * np.exp(-(noise ** 2) / (2.0 * sigma ** 2))

        h_cond, _ = integrate.quad(integrand, -20.0 * sigma, 20.0 * sigma, limit=200)
        capacities.append(1.0 - h_cond)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-1, 6), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.searchsorted(caps, rate)
    if idx == 0:
        return float(grid[0])
    if idx >= len(grid):
        return float(grid[-1])
    c0, c1 = caps[idx - 1], caps[idx]
    g0, g1 = grid[idx - 1], grid[idx]
    if c1 == c0:
        return float(g0)
    return float(g0 + (rate - c0) * (g1 - g0) / (c1 - c0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            rate = K_use / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
