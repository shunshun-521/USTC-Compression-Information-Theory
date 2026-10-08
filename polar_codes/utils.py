"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
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
    """从 CSV 加载仿真结果"""
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
    """BPSK 离散输入信道容量（bits/channel use）"""

    def cap_one(eb_n0_db):
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            t = -2.0 * snr * y
            log_term = np.logaddexp(0.0, t) / np.log(2.0)
            return log_term * np.exp(-0.5 * y ** 2)

        val, _ = integrate.quad(integrand, -np.inf, np.inf)
        val /= np.sqrt(2.0 * np.pi)
        return 1.0 - val

    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    return np.array([cap_one(e) for e in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-2, 15), num_points=400):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    caps_lo = compute_bpsk_capacity(lo, rate)
    caps_hi = compute_bpsk_capacity(hi, rate)
    if caps_lo < rate:
        while caps_lo < rate and lo > -10:
            lo -= 1.0
            caps_lo = compute_bpsk_capacity(lo, rate)
    if caps_hi > rate:
        while caps_hi > rate and hi < 30:
            hi += 1.0
            caps_hi = compute_bpsk_capacity(hi, rate)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if compute_bpsk_capacity(mid, rate) > rate:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


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
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={k_val / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
