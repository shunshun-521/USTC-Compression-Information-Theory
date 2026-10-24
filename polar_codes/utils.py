"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV。"""
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
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果。"""
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
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters") and row["avg_iters"]
                    else None,
                }
            )
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    BPSK-AWGN 信道容量（bits/channel use），
    采用等能量 BPSK 下 I(X;Y) 的常用高斯数值积分形式。
    """
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            z = -2.0 * snr * y * y
            if z < -50:
                t = 0.0
            else:
                t = np.log2(1.0 + np.exp(z))
            return t * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -20.0, 20.0)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(max(0.0, 1.0 - val))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    """找到使 BPSK 容量曲线与码率 R 相交的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    for i in range(1, len(eb_grid)):
        if (caps[i - 1] - rate) * (caps[i] - rate) <= 0:
            e0, e1 = eb_grid[i - 1], eb_grid[i]
            c0, c1 = caps[i - 1], caps[i]
            if c1 == c0:
                return float(e0)
            return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))
    snr_req = 2.0 ** rate - 1.0
    return float(10.0 * np.log10(snr_req / rate))


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
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, linewidth=1.5, markersize=5)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                k = N // 2
            else:
                k = K
            rate = k / N
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
