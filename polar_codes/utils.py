"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

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
        for row in results:
            writer.writerow(
                {
                    "eb_n0_db": row["eb_n0_db"],
                    "bler": row["bler"],
                    "ber": row["ber"],
                    "num_errors": row["num_errors"],
                    "num_frames": row["num_frames"],
                    "avg_decode_time_ms": row["avg_decode_time"] * 1000.0,
                    "avg_iters": "" if row["avg_iters"] is None else row["avg_iters"],
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_iters = row["avg_iters"]
            results.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": None if avg_iters == "" else float(avg_iters),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 二进制输入信道互信息（bits/channel use）。
    与 run_simulation 中 sigma 定义一致：sigma = 1/sqrt(2R*10^{Eb/N0/10})
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        sigma = 1.0 / np.sqrt(2.0 * rate * (10.0 ** (eb_n0_db / 10.0)))
        norm = np.sqrt(2.0 * np.pi) * sigma
        eps = 1e-300

        def integrand(y):
            p0y = np.exp(-0.5 * ((y - 1.0) / sigma) ** 2) / norm
            p1y = np.exp(-0.5 * ((y + 1.0) / sigma) ** 2) / norm
            py = 0.5 * p0y + 0.5 * p1y
            if py < eps:
                return 0.0
            term = 0.0
            for pyx in (p0y, p1y):
                if pyx > eps:
                    term += 0.5 * pyx * np.log2(pyx / py)
            return term

        val, _ = integrate.quad(integrand, -30.0, 30.0, limit=200)
        capacities.append(max(0.0, val))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = float(eb_n0_range[0]), float(eb_n0_range[1])
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if compute_bpsk_capacity(mid, rate) > rate:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线（semilogy）。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2, label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
