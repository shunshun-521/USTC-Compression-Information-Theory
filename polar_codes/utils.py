"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
                "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
            })


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


from channel import eb_n0_to_sigma


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 信道互信息（bits/channel use）。
    """
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    y = np.linspace(-10.0, 10.0, 20000)
    dy = y[1] - y[0]
    norm = sigma * np.sqrt(2.0 * np.pi)
    p0 = np.exp(-(y - 1.0) ** 2 / (2.0 * sigma ** 2)) / norm
    p1 = np.exp(-(y + 1.0) ** 2 / (2.0 * sigma ** 2)) / norm
    py = 0.5 * (p0 + p1)
    eps = 1e-300
    mi = 0.5 * np.sum(p0 * np.log2((p0 + eps) / (py + eps))) * dy
    mi += 0.5 * np.sum(p1 * np.log2((p1 + eps) / (py + eps))) * dy
    return max(0.0, mi)


def find_capacity_limit(rate, eb_n0_range=(-2, 8), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range
    eb_vals = np.linspace(lo, hi, num_points)
    caps = np.array([compute_bpsk_capacity(eb, rate) for eb in eb_vals])
    idx = np.searchsorted(caps, rate)
    if idx == 0:
        return lo
    if idx >= len(eb_vals):
        return hi
    e1, e2 = eb_vals[idx - 1], eb_vals[idx]
    c1, c2 = caps[idx - 1], caps[idx]
    if c2 == c1:
        return e1
    return e1 + (rate - c1) / (c2 - c1) * (e2 - e1)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

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
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
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
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
