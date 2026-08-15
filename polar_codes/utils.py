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
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000,
                    "avg_iters": (
                        float(row["avg_iters"]) if row["avg_iters"] else None
                    ),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    基于 LLR 的条件熵数值积分。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
        mean_llr = 2.0 * snr
        std_llr = np.sqrt(4.0 * snr)

        def integrand(llr):
            p0 = 1.0 / (1.0 + np.exp(-llr))
            p0 = np.clip(p0, 1e-15, 1.0 - 1e-15)
            h = -p0 * np.log2(p0) - (1.0 - p0) * np.log2(1.0 - p0)
            return h * np.exp(-0.5 * ((llr - mean_llr) / std_llr) ** 2) / (
                std_llr * np.sqrt(2.0 * np.pi)
            )

        upper = mean_llr + 12.0 * std_llr
        entropy, _ = integrate.quad(integrand, 0.0, upper, limit=200)
        capacities.append(max(0.0, 1.0 - entropy))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=200):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    """
    from scipy.optimize import brentq

    def objective(eb_db):
        return compute_bpsk_capacity([eb_db], rate)[0] - rate

    try:
        lo, hi = eb_n0_range
        cap_lo = compute_bpsk_capacity([lo], rate)[0]
        cap_hi = compute_bpsk_capacity([hi], rate)[0]
        if cap_lo > rate:
            return lo
        if cap_hi < rate:
            return hi
        return brentq(objective, lo, hi)
    except ValueError:
        # 文献常用近似值（R=1/2 BPSK 香农限约 0.188 dB）
        return 0.188 if abs(rate - 0.5) < 1e-6 else 0.0


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
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, results in results_dict.items():
        eb_n0 = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb_n0, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(
            x=shannon_limit_db,
            color="gray",
            linestyle="--",
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
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
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(
                N, K_val, design_eb_n0_db, rate=rate
            )
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
