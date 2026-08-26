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
    """将仿真结果保存为 CSV 文件。"""
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
        for r in results:
            writer.writerow(
                {
                    "eb_n0_db": f"{r['eb_n0_db']:.4f}",
                    "bler": f"{r['bler']:.6e}",
                    "ber": f"{r['ber']:.6e}",
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": f"{r['avg_decode_time'] * 1000:.6f}",
                    "avg_iters": (
                        f"{r['avg_iters']:.4f}" if r["avg_iters"] is not None else ""
                    ),
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
                    "avg_iters": (
                        float(row["avg_iters"]) if row["avg_iters"].strip() else None
                    ),
                }
            )
    return results


def bpsk_capacity_per_use(snr_linear):
    """BPSK 离散输入信道容量（bits/channel use）。"""

    def integrand(y):
        t = -snr_linear * y ** 2
        if t > 50:
            log_term = 0.0
        elif t < -50:
            log_term = -t / np.log(2)
        else:
            log_term = np.log2(1.0 + np.exp(t))
        return log_term * np.exp(-0.5 * y ** 2) / np.sqrt(2.0 * np.pi)

    val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算给定 Eb/N0 列表下的 BPSK 信道容量。"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        caps.append(bpsk_capacity_per_use(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 8), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        snr = 2.0 * rate * (10 ** (mid / 10.0))
        cap = bpsk_capacity_per_use(snr)
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    for idx, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(
            eb,
            bler,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            markersize=5,
            label=label,
        )
    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
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
            f.write(np.array2string(info_idx, threshold=N, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, separator=" ") + "\n")
            f.write("-" * 53 + "\n")
