"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate

from construction import ga_construction


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
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                f"{avg_iters:.2f}" if avg_iters is not None else "",
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
                "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
            })
    return results


def _bpsk_mi_integrand(y, sigma):
    norm = np.sqrt(2.0 * np.pi) * sigma
    p0 = np.exp(-((y - 1.0) ** 2) / (2.0 * sigma ** 2)) / norm
    p1 = np.exp(-((y + 1.0) ** 2) / (2.0 * sigma ** 2)) / norm
    py = 0.5 * (p0 + p1)
    val = 0.0
    for p_yx in (p0, p1):
        if p_yx > 1e-300 and py > 1e-300:
            val += 0.5 * p_yx * np.log2(p_yx / py)
    return val


def compute_bpsk_capacity(eb_n0_db_list, rate=None):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    C = I(X;Y) for BPSK over AWGN，与码率无关。
    """
    caps = []
    for eb_n0_db in eb_n0_db_list:
        sigma = 1.0 / np.sqrt(2.0 * (10 ** (eb_n0_db / 10.0)))
        mi, _ = integrate.quad(lambda y: _bpsk_mi_integrand(y, sigma), -30, 30, limit=200)
        caps.append(max(mi, 0.0))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    这是香农限，用于在 BLER 图中标注参考竖线。
    """
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.searchsorted(caps, rate)
    if idx == 0:
        return float(eb_grid[0])
    if idx >= len(eb_grid):
        return float(eb_grid[-1])
    c0, c1 = caps[idx - 1], caps[idx]
    e0, e1 = eb_grid[idx - 1], eb_grid[idx]
    if abs(c1 - c0) < 1e-12:
        return float((e0 + e1) / 2.0)
    return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, marker=markers[idx % len(markers)], label=label, linewidth=1.5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Capacity limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=9)
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
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
