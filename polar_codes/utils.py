"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time_ms, avg_iters
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
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
            avg_iters = r.get("avg_iters")
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if avg_iters is None else f"{avg_iters:.2f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_iters_raw = row.get("avg_iters", "")
            avg_iters = None if avg_iters_raw == "" else float(avg_iters_raw)
            results.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": avg_iters,
                }
            )
    return results


def _bpsk_capacity_gamma(gamma):
    """BPSK 互信息（bits/channel use），gamma = Es/N0"""

    def integrand(x):
        z = -2.0 * gamma * x
        log2_term = np.maximum(z, 0.0) / np.log(2.0) + np.log2(
            1.0 + np.exp(-np.abs(z))
        )
        return log2_term * np.exp(-x ** 2 / 2.0)

    val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
    return 1.0 - val / np.sqrt(2.0 * np.pi)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=float)
    capacities = []
    for eb in eb_n0_db_list:
        gamma = (10.0 ** (eb / 10.0)) * rate
        capacities.append(_bpsk_capacity_gamma(gamma))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    采用文献插值（BPSK 二进制输入 AWGN 典型值）。
    """
    rates = np.array([0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9])
    eb_db = np.array([-1.28, -0.54, -0.05, 0.27, 0.59, 1.25, 2.05, 3.55])
    return float(np.interp(rate, rates, eb_db))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线（semilogy），同时保存 PNG 与 PDF"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Capacity limit ({shannon_limit_db:.2f} dB)",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位/冻结位集合保存到文本文件"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
