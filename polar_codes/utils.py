"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
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
            writer.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    "" if r.get("avg_iters") is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK-AWGN 互信息（bits/channel use），与仿真信道参数一致"""
    y = np.linspace(-8.0, 8.0, 16001)
    dy = y[1] - y[0]
    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
        sigma = 1.0 / np.sqrt(snr)
        p0 = np.exp(-0.5 * ((y - 1.0) / sigma) ** 2)
        p1 = np.exp(-0.5 * ((y + 1.0) / sigma) ** 2)
        py = 0.5 * (p0 + p1)
        py /= np.sum(py) * dy
        hy = -np.sum(py * np.log2(py + 1e-300)) * dy
        hy_given_x = 0.5 * np.log2(2.0 * np.pi * np.e * sigma**2)
        caps.append(max(hy - hy_given_x, 0.0))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-3.0, 6.0)):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    for _ in range(60):
        mid = (lo + hi) / 2.0
        c = compute_bpsk_capacity([mid], rate)[0]
        if c > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(x, y, "o-", label=label, markersize=4)
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
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = K if K is not None else N // 2
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
