"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
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
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    r["avg_iters"] if r["avg_iters"] is not None else "",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
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
                    "avg_iters": (
                        float(row["avg_iters"]) if row["avg_iters"] else None
                    ),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use），蒙特卡洛估计。
    """
    sigma = 1.0 / np.sqrt(2.0 * rate * (10 ** (eb_n0_db / 10.0)))
    rng = np.random.default_rng(0)
    y = rng.normal(1.0, sigma, size=200000)
    llr = 2.0 * y / (sigma ** 2)
    t = np.where(
        llr >= 0,
        np.log1p(np.exp(-llr)),
        np.log1p(np.exp(llr)) - llr,
    )
    return 1.0 - np.mean(t) / np.log(2)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([compute_bpsk_capacity(db, rate) for db in grid])
    for i in range(len(grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i])
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    return float(grid[np.argmin(np.abs(caps - rate))])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Capacity limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_arg, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K = K_arg if K_arg is not None else N // 2
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N) + "\n")
            f.write("-" * 53 + "\n")
