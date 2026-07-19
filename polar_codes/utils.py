"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
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
        for r in results:
            writer.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": "" if r["avg_iters"] is None else r["avg_iters"],
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
                    "avg_iters": (
                        None
                        if row["avg_iters"] == "" or row["avg_iters"] is None
                        else float(row["avg_iters"])
                    ),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use，离散输入数值积分）。
    """
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64).ravel()
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        es_n0 = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(x):
            # 对称 BPSK 互信息：1 - E[log2(1+exp(-2*Es/N0*x^2))]
            arg = -2.0 * es_n0 * (x ** 2)
            if arg < -60:
                penalty = 0.0
            elif arg > 60:
                penalty = arg / np.log(2)
            else:
                penalty = np.log2(1.0 + np.exp(arg))
            return penalty * np.exp(-(x ** 2)) / np.sqrt(np.pi)

        val, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
        capacities.append(max(0.0, min(1.0, 1.0 - val)))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 10), num_points=2000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    若数值积分未找到交点，则回退到连续输入 AWGN 香农限。
    """
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    diff = caps - rate
    for i in range(len(eb_grid) - 1):
        if diff[i] * diff[i + 1] <= 0:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i] + 1e-15)
            return float(eb_grid[i] + t * (eb_grid[i + 1] - eb_grid[i]))
    # 连续输入 AWGN 香农限（常用参考）
    snr_linear = 2.0 ** (2.0 * rate) - 1.0
    return float(10.0 * np.log10(snr_linear / (2.0 * rate)))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(
            eb,
            bler,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            markersize=5,
            label=label,
        )

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
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
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
            k_val = N // 2 if K is None else K
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
