"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
                    "avg_iters": None
                    if row["avg_iters"] in ("", "None", None)
                    else float(row["avg_iters"]),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate=None):
    """
    计算 BPSK-AWGN 信道互信息 I(X;Y)（bits/channel use）。
    """
    from scipy.stats import norm

    _ = rate  # 保留接口兼容
    capacities = []
    y_grid = np.linspace(-12, 12, 12000)
    dy = y_grid[1] - y_grid[0]

    for eb_n0_db in eb_n0_db_list:
        eb_n0 = 10.0 ** (eb_n0_db / 10.0)
        sigma = 1.0 / np.sqrt(2.0 * eb_n0)
        p_y_given_m1 = norm.pdf(y_grid, loc=1.0, scale=sigma)
        p_y_given_p1 = norm.pdf(y_grid, loc=-1.0, scale=sigma)
        p_y = 0.5 * (p_y_given_m1 + p_y_given_p1)
        h_y = -np.sum(p_y * np.log2(p_y + 1e-300)) * dy
        h_y_given_x = 0.5 * np.log2(2.0 * np.pi * np.e * sigma ** 2)
        capacities.append(max(0.0, float(h_y - h_y_given_x)))

    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=1000):
    """找到使 BPSK 信道互信息等于码率 R 的 Eb/N0（dB）。"""
    eb_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_vals)
    idx = np.where(caps >= rate)[0]
    if len(idx) == 0:
        return float(eb_vals[-1])
    if idx[0] == 0:
        return float(eb_vals[0])

    lo = eb_vals[idx[0] - 1]
    hi = eb_vals[idx[0]]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        cap = compute_bpsk_capacity([mid])[0]
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = N // 2 if K is None else K
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
