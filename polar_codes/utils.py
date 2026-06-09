"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.2f}",
                ]
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
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters", "").strip()
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    通过 BPSK-AWGN 互信息数值积分：I = 1 - H(X|Y)。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb_lin = 10 ** (eb_n0_db / 10.0)
        snr = 2.0 * rate * eb_lin
        sigma = 1.0 / np.sqrt(snr) if snr > 0 else 1e9

        def p0(y):
            return np.exp(-((y - 1.0) ** 2) / (2.0 * sigma**2)) / (
                sigma * np.sqrt(2.0 * np.pi)
            )

        def p1(y):
            return np.exp(-((y + 1.0) ** 2) / (2.0 * sigma**2)) / (
                sigma * np.sqrt(2.0 * np.pi)
            )

        def integrand(y):
            p = 0.5 * p0(y) + 0.5 * p1(y)
            if p < 1e-300:
                return 0.0
            p0y = 0.5 * p0(y) / p
            p1y = 0.5 * p1(y) / p
            h = 0.0
            for q in (p0y, p1y):
                if q > 1e-15:
                    h -= q * np.log2(q)
            return p * h

        cond_entropy, _ = integrate.quad(integrand, -12.0, 12.0, limit=200)
        capacities.append(1.0 - cond_entropy)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 10), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    for _ in range(60):
        mid = (lo + hi) / 2.0
        c = compute_bpsk_capacity([mid], rate)[0]
        if c < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线（semilogy），同时保存 PNG 和 PDF。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=5)

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
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位/冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k = N // 2 if K is None else K
            rate = k / N
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(" ".join(f"{i:4d}" for i in info_idx) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(" ".join(f"{i:4d}" for i in frozen_idx) + "\n")
            f.write("-" * 53 + "\n")
