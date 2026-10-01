"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from construction import ga_construction


def save_results_csv(results, filepath):
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
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000,
                    r.get("avg_iters", "") if r.get("avg_iters") is not None else "",
                ]
            )


def load_results_csv(filepath):
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
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000,
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    BPSK 离散输入互信息（bits/channel use），蒙特卡洛估计。
    """
    eb = 10 ** (eb_n0_db / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * rate * eb))
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, sigma, size=200000)
    llr = 2.0 * y / (sigma ** 2)
    p0 = 1.0 / (1.0 + np.exp(-llr))
    p0 = np.clip(p0, 1e-12, 1 - 1e-12)
    h_cond = -(p0 * np.log2(p0) + (1 - p0) * np.log2(1 - p0))
    return float(1.0 - np.mean(h_cond))


def find_capacity_limit(rate, eb_n0_range=(-5, 10), num_points=200):
    lo, hi = eb_n0_range
    grid = np.linspace(lo, hi, num_points)
    caps = [compute_bpsk_capacity(eb, rate) for eb in grid]
    for i in range(1, len(grid)):
        if caps[i - 1] < rate <= caps[i]:
            return float(grid[i])
    if caps[-1] >= rate:
        return float(grid[-1])
    return float(hi)


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        plt.semilogy(xs, ys, marker="o", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            rate = K_use / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_use}):\n{np.array2string(info_idx, threshold=N)}\n")
            f.write(
                f"Frozen indices (all {N-K_use}):\n{np.array2string(frozen_idx, threshold=N)}\n"
            )
            f.write("-" * 53 + "\n")
