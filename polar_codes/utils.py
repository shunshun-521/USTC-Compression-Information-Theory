"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy.stats import norm

from channel import eb_n0_to_sigma
from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV"""
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = dict(r)
            row["avg_decode_time"] = row["avg_decode_time"] * 1000.0
            if row["avg_iters"] is None:
                row["avg_iters"] = ""
            w.writerow(row)


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    out = []
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
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db, rate):
    """BPSK-AWGN 对称信道容量（bits/channel use）"""
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    ys = np.linspace(-1.0 - 10.0 * sigma, 1.0 + 10.0 * sigma, 8001)
    p_pos = norm.pdf(ys - 1.0, scale=sigma)
    p_neg = norm.pdf(ys + 1.0, scale=sigma)
    denom = np.maximum(p_pos + p_neg, 1e-300)
    mi = 0.5 * np.trapezoid(p_pos * np.log2(2.0 * p_pos / denom), ys)
    mi += 0.5 * np.trapezoid(p_neg * np.log2(2.0 * p_neg / denom), ys)
    return float(np.clip(mi, 0.0, 1.0))


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=200):
    """找到 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    c_lo = compute_bpsk_capacity(lo, rate) - rate
    c_hi = compute_bpsk_capacity(hi, rate) - rate
    if c_lo * c_hi > 0:
        grid = np.linspace(lo, hi, num_points)
        caps = np.array([compute_bpsk_capacity(eb, rate) for eb in grid])
        return float(grid[np.argmin(np.abs(caps - rate))])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if compute_bpsk_capacity(mid, rate) < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER 曲线（PNG + PDF）"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    base, _ = os.path.splitext(save_path)
    plt.savefig(base + ".pdf")
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate=K_use / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={K_use / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
