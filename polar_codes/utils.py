"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time, avg_iters
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            avg_iters = r.get("avg_iters")
            writer.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                "" if avg_iters is None else f"{avg_iters:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
            })
    return results


def _bpsk_capacity_per_snr(snr_linear):
    """BPSK 离散输入信道容量（bits/channel use），snr_linear = Es/N0。"""
    y = np.linspace(-15.0, 15.0, 20001)
    dy = y[1] - y[0]
    p0 = np.exp(-0.5 * (y - np.sqrt(snr_linear)) ** 2) / np.sqrt(2.0 * np.pi)
    p1 = np.exp(-0.5 * (y + np.sqrt(snr_linear)) ** 2) / np.sqrt(2.0 * np.pi)
    py = np.clip(0.5 * p0 + 0.5 * p1, 1e-300, None)
    hy = -np.sum(py * np.log2(py) * dy)
    hyc = 0.5 * np.log2(2.0 * np.pi * np.e)
    return hy - hyc


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        es_n0 = 2.0 * rate * 10 ** (eb_n0_db / 10.0)
        capacities.append(_bpsk_capacity_per_snr(es_n0))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=1000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    这是香农限，用于在 BLER 图中标注参考竖线。
    """
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_n0_vals, rate)

    for i in range(len(caps) - 1):
        if caps[i] <= rate <= caps[i + 1]:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i])
            return eb_n0_vals[i] + t * (eb_n0_vals[i + 1] - eb_n0_vals[i])

    if caps[-1] < rate:
        return eb_n0_range[1]
    return eb_n0_range[0]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, results in results_dict.items():
        eb_n0 = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb_n0, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Capacity limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = N // 2 if K is None else K
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)

            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
