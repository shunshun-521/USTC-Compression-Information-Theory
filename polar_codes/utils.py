"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV"""
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
                "avg_decode_time",
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
                    r["avg_decode_time"],
                    "" if r.get("avg_iters") is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"])
            it = row.get("avg_iters", "")
            row["avg_iters"] = None if it == "" or it is None else float(it)
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))

        def integrand(y):
            t = -2.0 * snr * (y ** 2)
            if t > 0:
                log_term = t / np.log(2.0) + np.log1p(np.exp(-t)) / np.log(2.0)
            else:
                log_term = np.log1p(np.exp(t)) / np.log(2.0)
            return log_term * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(grid) - 1:
        return float(grid[idx])
    # 线性插值
    for i in range(len(grid) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0:
            t = (rate - c0) / (c1 - c0 + 1e-12)
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    return float(grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, marker="o", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_eff = K if K is not None else N // 2
            rate = K_eff / N
            info, frozen, _ = ga_construction(N, K_eff, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_eff}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n")
            f.write(np.array2string(info, threshold=info.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n")
            f.write(np.array2string(frozen, threshold=frozen.size) + "\n")
            f.write("-" * 53 + "\n")
