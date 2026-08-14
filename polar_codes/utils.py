"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import brentq

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
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "eb_n0_db": r["eb_n0_db"],
                "bler": r["bler"],
                "ber": r["ber"],
                "num_errors": r["num_errors"],
                "num_frames": r["num_frames"],
                "avg_decode_time_ms": r["avg_decode_time"] * 1000,
                "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
            })


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000,
                "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    通过输出分布微分熵数值计算。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb_lin = 10 ** (eb_n0_db / 10.0)
        snr = 2 * rate * eb_lin
        sigma = 1.0 / np.sqrt(snr)
        norm = sigma * np.sqrt(2 * np.pi)

        def p_y(y):
            return 0.5 * (
                np.exp(-((y - 1) ** 2) / (2 * sigma ** 2))
                + np.exp(-((y + 1) ** 2) / (2 * sigma ** 2))
            ) / norm

        def entropy_y():
            def integrand(y):
                py = p_y(y)
                if py < 1e-300:
                    return 0.0
                return -py * np.log2(py)

            val, _ = integrate.quad(integrand, -20, 20, limit=200)
            return val

        h_noise = 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2)
        capacities.append(entropy_y() - h_noise)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""

    def cap_minus_rate(eb_n0_db):
        return compute_bpsk_capacity([eb_n0_db], rate)[0] - rate

    lo, hi = eb_n0_range
    flo, fhi = cap_minus_rate(lo), cap_minus_rate(hi)
    if flo * fhi > 0:
        eb_grid = np.linspace(lo, hi, num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        return float(eb_grid[np.argmin(np.abs(caps - rate))])
    return float(brentq(cap_minus_rate, lo, hi))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(
            eb, bler,
            marker=markers[i % len(markers)],
            label=label,
            linewidth=1.5,
        )

    if shannon_limit_db is not None:
        ax.axvline(
            x=shannon_limit_db,
            color="gray",
            linestyle="--",
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
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位/冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
