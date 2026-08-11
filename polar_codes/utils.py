"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
        "avg_decode_time_ms", "avg_iters",
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
                "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    snr = 2.0 * rate * 10.0 ** (eb_n0_db / 10.0)

    def integrand(y):
        return np.log2(1.0 + np.exp(-2.0 * snr * y)) * np.exp(-y ** 2 / 2.0)

    val, _ = integrate.quad(integrand, -np.inf, np.inf)
    val /= np.sqrt(2.0 * np.pi)
    return 1.0 - val


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = [compute_bpsk_capacity(eb, rate) for eb in eb_n0_vals]
    caps = np.array(caps)
    idx = np.argmin(np.abs(caps - rate))
    if idx > 0 and idx < len(eb_n0_vals) - 1:
        e0, e1 = eb_n0_vals[idx - 1], eb_n0_vals[idx + 1]
        c0 = compute_bpsk_capacity(e0, rate)
        c1 = compute_bpsk_capacity(e1, rate)
        if abs(c1 - c0) > 1e-12:
            return e0 + (rate - c0) / (c1 - c0) * (e1 - e0)
    return eb_n0_vals[idx]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, marker=markers[i % len(markers)],
                     label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--",
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = int(K) if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
