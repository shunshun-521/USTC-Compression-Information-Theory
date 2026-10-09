"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

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
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
    results = []
    with open(filepath, newline="") as f:
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def _log2_one_plus_exp(x):
    """数值稳定的 log2(1 + exp(x))。"""
    if x < -40.0:
        return 0.0
    if x > 40.0:
        return x / np.log(2.0)
    return np.log1p(np.exp(x)) / np.log(2.0)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            return _log2_one_plus_exp(-2.0 * snr * y) * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -12.0, 12.0)
        val /= np.sqrt(2.0 * np.pi)
        capacities.append(1.0 - val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    c_lo = compute_bpsk_capacity([lo], rate)[0]
    c_hi = compute_bpsk_capacity([hi], rate)[0]
    if c_lo < rate:
        lo = -5.0
    if c_hi > rate:
        hi = 15.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        c_mid = compute_bpsk_capacity([mid], rate)[0]
        if c_mid < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


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
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(x, y, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
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
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
