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
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
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
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters") not in (None, "")
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    I(X;Y) = 1 - H(Y|X)，SNR = 2R * 10^{Eb/N0/10}。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        gamma = 2.0 * rate * (10 ** (eb_n0_db / 10.0))

        def log2_one_plus_exp(z):
            z = float(z)
            if z > 0:
                return z / np.log(2) + np.log2(1.0 + np.exp(-z))
            return np.log2(1.0 + np.exp(z))

        def integrand(x):
            t1 = np.exp(-((x - 1.0) ** 2) / 2.0) * log2_one_plus_exp(-2.0 * gamma * x)
            t2 = np.exp(-((x + 1.0) ** 2) / 2.0) * log2_one_plus_exp(2.0 * gamma * x)
            return 0.5 * (t1 + t2) / np.sqrt(2.0 * np.pi)

        h_yx, _ = integrate.quad(integrand, -20.0, 20.0)
        capacities.append(1.0 - h_yx)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 10), num_points=2000):
    """找到使 BPSK 信道容量最接近码率 R 的 Eb/N0（dB）。"""
    from scipy.optimize import minimize_scalar

    def objective(eb_db):
        cap = compute_bpsk_capacity([eb_db], rate)[0]
        return (cap - rate) ** 2

    result = minimize_scalar(objective, bounds=eb_n0_range, method="bounded")
    return float(result.x)


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
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
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
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
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
            f.write(np.array2string(info_idx, threshold=len(info_idx)) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=len(frozen_idx)) + "\n")
            f.write("-" * 53 + "\n")
