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
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["avg_decode_time"] = r["avg_decode_time"] * 1000.0
            if row["avg_iters"] is None:
                row["avg_iters"] = ""
            writer.writerow(row)


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"]) / 1000.0
            row["avg_iters"] = float(row["avg_iters"]) if row["avg_iters"] else None
            results.append(row)
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""

    def capacity_at_eb_n0(eb_n0_db):
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
        log2 = np.log(2.0)

        def integrand(y):
            # y ~ N(0,1)，使用 |y| 保证期望有限
            t = -2.0 * snr * np.abs(y)
            log_term = np.logaddexp(0.0, t) / log2
            return log_term * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        return 1.0 - val / np.sqrt(2.0 * np.pi)

    if np.isscalar(eb_n0_db_list):
        return capacity_at_eb_n0(eb_n0_db_list)
    return np.array([capacity_at_eb_n0(e) for e in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    from scipy.optimize import brentq

    lo, hi = eb_n0_range[0], eb_n0_range[1]
    caps_lo = compute_bpsk_capacity(lo, rate)
    caps_hi = compute_bpsk_capacity(hi, rate)
    if caps_lo > rate:
        return float(lo)
    if caps_hi < rate:
        return float(hi)

    return float(brentq(lambda eb: compute_bpsk_capacity(eb, rate) - rate, lo, hi))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(xs, ys, marker="o", linewidth=1.5, label=label)

    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_n}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {N - K_n}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
