"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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
    """从 CSV 文件加载仿真结果"""
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def _stable_log2_one_plus_exp(x):
    """数值稳定的 log2(1 + exp(x))，支持数组"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = x[pos] / np.log(2) + np.log2(1 + np.exp(-x[pos]))
    out[~pos] = np.log2(1 + np.exp(x[~pos]))
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate, n_samples=80000):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use）。
    通过蒙特卡洛估计互信息 I(X;Y)。
    """
    rng = np.random.default_rng(0)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb_lin = 10 ** (eb_n0_db / 10)
        sigma = 1.0 / np.sqrt(2 * rate * eb_lin) if eb_lin > 0 else 1e3
        bits = rng.integers(0, 2, size=n_samples)
        x = 1 - 2 * bits
        y = x + rng.normal(0, sigma, size=n_samples)
        p0 = np.exp(-0.5 * ((y - 1) / sigma) ** 2)
        p1 = np.exp(-0.5 * ((y + 1) / sigma) ** 2)
        py = 0.5 * (p0 + p1)
        px0y = 0.5 * p0 / py
        px1y = 0.5 * p1 / py
        h = np.zeros(n_samples)
        h += np.where(px0y > 1e-15, -px0y * np.log2(px0y), 0.0)
        h += np.where(px1y > 1e-15, -px1y * np.log2(px1y), 0.0)
        capacities.append(float(1.0 - np.mean(h)))
    return np.array(capacities)


def find_capacity_limit(rate):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = -2.0, 2.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if compute_bpsk_capacity([mid], rate, n_samples=40000)[0] > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
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
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
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
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
