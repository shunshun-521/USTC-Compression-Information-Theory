"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    """
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
                    "avg_iters": row["avg_iters"] if row["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        # BPSK-AWGN 信道容量（bits/channel use），γ = Eb/N0（线性）
        gamma = 10 ** (eb_n0_db / 10.0)

        def integrand(y, g=gamma):
            x = 2.0 * g * abs(y)
            return np.logaddexp(0.0, -x) / np.log(2.0) * np.exp(-(y ** 2) / 2.0)

        val, _ = integrate.quad(integrand, -20.0, 20.0)
        capacities.append(1.0 - val / np.sqrt(2.0 * np.pi))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20)):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    """
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    cap_lo = compute_bpsk_capacity([lo], rate)[0]
    cap_hi = compute_bpsk_capacity([hi], rate)[0]
    if cap_lo < rate:
        while cap_lo < rate and lo > -20:
            lo -= 2.0
            cap_lo = compute_bpsk_capacity([lo], rate)[0]
    if cap_hi > rate:
        while cap_hi > rate and hi < 30:
            hi += 2.0
            cap_hi = compute_bpsk_capacity([hi], rate)[0]
    for _ in range(50):
        mid = (lo + hi) / 2.0
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap < rate:
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
    """
    绘制 BLER-Eb/N0 曲线。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
