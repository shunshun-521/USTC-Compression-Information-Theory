"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

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
                    "avg_iters": "" if r["avg_iters"] is None else r["avg_iters"],
                }
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
                    if row["avg_iters"] in ("", None)
                    else float(row["avg_iters"]),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate, num_samples=200000):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    使用蒙特卡洛估计互信息 I(X;Y)。
    """
    rng = np.random.default_rng(0)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        a = np.sqrt(2.0 * rate * (10.0 ** (eb_n0_db / 10.0)))
        x = rng.choice([-1, 1], size=num_samples)
        y = a * x + rng.normal(0.0, 1.0, size=num_samples)
        p1 = 1.0 / (1.0 + np.exp(-a * y))
        h = -p1 * np.log2(p1 + 1e-12) - (1.0 - p1) * np.log2(1.0 - p1 + 1e-12)
        capacities.append(1.0 - float(np.mean(h)))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate, num_samples=50000)
    idx = np.searchsorted(caps, rate)
    idx = min(max(idx, 1), len(caps) - 1)
    c0, c1 = caps[idx - 1], caps[idx]
    e0, e1 = eb_grid[idx - 1], eb_grid[idx]
    if abs(c1 - c0) < 1e-12:
        return float(e0)
    t = (rate - c0) / (c1 - c0)
    return float(e0 + t * (e1 - e0))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(x, y, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                K_val = N // 2
            else:
                K_val = K
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_val}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - K_val}):\n")
            f.write(
                np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n"
            )
            f.write("-" * 53 + "\n")
