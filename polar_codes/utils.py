"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from construction import ga_construction


def save_results_csv(results, filepath):
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
                    if row.get("avg_iters") and row["avg_iters"] != ""
                    else None,
                }
            )
    return results


def _bpsk_mutual_info(eb_n0_db, rate):
    """BPSK-AWGN 互信息（bits/channel use），与 channel.py 的 SNR 定义一致"""
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    sigma = 1.0 / np.sqrt(snr)
    grid = np.linspace(-12.0 * sigma, 12.0 * sigma, 12001)
    dy = grid[1] - grid[0]
    p_y = (
        np.exp(-((grid - 1.0) ** 2) / (2.0 * sigma ** 2))
        + np.exp(-((grid + 1.0) ** 2) / (2.0 * sigma ** 2))
    ) / (2.0 * np.sqrt(2.0 * np.pi * sigma ** 2))
    p_y = np.clip(p_y, 1e-300, None)
    h_y = -np.sum(p_y * np.log2(p_y)) * dy
    h_y_given_x = 0.5 * np.log2(2.0 * np.pi * np.e * sigma ** 2)
    return h_y - h_y_given_x


def compute_bpsk_capacity(eb_n0_db_list, rate):
    return np.array([_bpsk_mutual_info(eb, rate) for eb in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    for i in range(len(grid) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0 and c1 != c0:
            t = (rate - c0) / (c1 - c0)
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    idx = int(np.argmin(np.abs(caps - rate)))
    return float(grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(x, y, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {k_val}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - k_val}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
