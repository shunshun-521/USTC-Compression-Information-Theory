"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            writer.writerow([
                r["eb_n0_db"],
                r["bler"],
                r["ber"],
                r["num_errors"],
                r["num_frames"],
                r["avg_decode_time"] * 1000,
                "" if r["avg_iters"] is None else r["avg_iters"],
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
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
                "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 离散输入信道容量（bits/channel use）。
    C = 1 - E_y[log2(1 + exp(-2*Eb/N0 * y^2))], y ~ N(0,1)
    其中 Eb/N0 为线性值。
    """
    eb_n0 = 10.0 ** (eb_n0_db / 10.0)

    def integrand(y):
        arg = -2.0 * eb_n0 * y ** 2
        log_term = np.where(arg < -50, 0.0, np.log2(1.0 + np.exp(arg)))
        return log_term * np.exp(-0.5 * y ** 2)

    val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
    val /= np.sqrt(2.0 * np.pi)
    return max(0.0, 1.0 - val)


def find_capacity_limit(rate, eb_n0_range=(-2, 6)):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    if compute_bpsk_capacity(lo, rate) < rate:
        while lo > -5 and compute_bpsk_capacity(lo, rate) < rate:
            lo -= 0.5
    if compute_bpsk_capacity(hi, rate) > rate:
        while hi < 15 and compute_bpsk_capacity(hi, rate) > rate:
            hi += 0.5

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if compute_bpsk_capacity(mid, rate) >= rate:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线（PNG + PDF）"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

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
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator="  ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator="  ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
