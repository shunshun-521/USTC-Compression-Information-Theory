"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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
            avg_iters = "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}"
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    avg_iters,
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
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
                    if row["avg_iters"].strip()
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 对称信道容量（bits/channel use）。
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = rate * (10.0 ** (eb_n0_db / 10.0))
        sigma2 = 1.0 / (2.0 * snr)
        scale = np.sqrt(2.0 * np.pi * sigma2)

        def integrand(y):
            p0 = np.exp(-((y - 1.0) ** 2) / (2.0 * sigma2))
            p1 = np.exp(-((y + 1.0) ** 2) / (2.0 * sigma2))
            s = p0 + p1
            t0 = np.where(p0 > 1e-300, p0 * np.log2(2.0 * p0 / s), 0.0)
            t1 = np.where(p1 > 1e-300, p1 * np.log2(2.0 * p1 / s), 0.0)
            return (t0 + t1) / (2.0 * scale)

        val, _ = integrate.quad(integrand, -20.0, 20.0)
        capacities.append(val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-1.5, 3.0), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.where(np.diff(np.sign(caps - rate)))[0]
    if len(idx) == 0:
        return float(eb_grid[np.argmin(np.abs(caps - rate))])
    i = idx[0]
    x0, x1 = eb_grid[i], eb_grid[i + 1]
    c0, c1 = caps[i], caps[i + 1]
    if c1 == c0:
        return float(x0)
    return float(x0 + (rate - c0) * (x1 - x0) / (c1 - c0))


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
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

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


def save_frozen_set_info(N_list, K_default, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K = K_default if K_default is not None else N // 2
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ") + "\n")
            f.write("-" * 53 + "\n")
