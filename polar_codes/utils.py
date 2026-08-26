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
        for row in results:
            writer.writerow(
                {
                    "eb_n0_db": row["eb_n0_db"],
                    "bler": row["bler"],
                    "ber": row["ber"],
                    "num_errors": row["num_errors"],
                    "num_frames": row["num_frames"],
                    "avg_decode_time_ms": row["avg_decode_time"] * 1000.0,
                    "avg_iters": row["avg_iters"] if row["avg_iters"] is not None else "",
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    使用 BPSK-AWGN 互信息数值积分（与 channel.py 的 sigma 定义一致）。
    """
    sigma = 1.0 / np.sqrt(2.0 * rate * (10 ** (eb_n0_db / 10.0)))
    y = np.linspace(-12.0, 12.0, 24001)
    dy = y[1] - y[0]

    def sym_pdf(y_val, x_bit):
        s = 1.0 - 2.0 * x_bit
        return np.exp(-((y_val - s) ** 2) / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)

    p_y = 0.5 * sym_pdf(y, 0) + 0.5 * sym_pdf(y, 1)
    mi = 0.0
    for x_bit in (0, 1):
        p_yx = sym_pdf(y, x_bit)
        mask = p_yx > 1e-300
        ratio = np.zeros_like(y)
        ratio[mask] = p_yx[mask] / p_y[mask]
        integrand = np.zeros_like(y)
        integrand[mask] = p_yx[mask] * np.log2(ratio[mask])
        mi += 0.5 * np.trapezoid(integrand, y)
    return max(0.0, mi)


def find_capacity_limit(rate, eb_n0_range=(-5.0, 20.0), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([compute_bpsk_capacity(eb, rate) for eb in eb_grid])
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        return float(eb_grid[idx])
    x0, x1 = eb_grid[idx - 1], eb_grid[idx + 1]
    c0 = compute_bpsk_capacity(x0, rate) - rate
    c1 = compute_bpsk_capacity(x1, rate) - rate
    if c0 * c1 > 0:
        return float(eb_grid[idx])
    for _ in range(50):
        mid = (x0 + x1) / 2.0
        cm = compute_bpsk_capacity(mid, rate) - rate
        if c0 * cm <= 0:
            x1, c1 = mid, cm
        else:
            x0, c0 = mid, cm
    return (x0 + x1) / 2.0


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
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_or_rate, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K_or_rate is None:
                K = N // 2
            elif isinstance(K_or_rate, float):
                K = int(N * K_or_rate)
            else:
                K = K_or_rate
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db, rate)

            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
