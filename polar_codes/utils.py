"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
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
            w.writerow(
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
                    if row.get("avg_iters") not in (None, "")
                    else None,
                }
            )
    return results


def _bpsk_capacity_scalar(snr_linear):
    """BPSK-AWGN 互信息（与 channel.py LLR 定义一致）"""
    sigma2 = 1.0 / snr_linear
    sigma = np.sqrt(sigma2)

    def branch(mean, x_bit):
        def integrand(y):
            llr = 2.0 * y / sigma2
            z = (1 - 2 * x_bit) * llr
            if z > 0:
                term = np.log2(1.0 + np.exp(-z))
            else:
                term = np.log2(1.0 + np.exp(z)) - z / np.log(2)
            return term * np.exp(-0.5 * ((y - mean) ** 2) / sigma2) / (
                np.sqrt(2.0 * np.pi) * sigma
            )

        val, _ = integrate.quad(integrand, -50.0, 50.0, limit=200)
        return val

    h_cond = 0.5 * (branch(1.0, 0) + branch(-1.0, 1))
    return 1.0 - h_cond


def compute_bpsk_capacity(eb_n0_db_list, rate):
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        caps.append(_bpsk_capacity_scalar(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=1000):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    for i in range(1, len(grid)):
        if caps[i - 1] < rate <= caps[i]:
            t = (rate - caps[i - 1]) / (caps[i] - caps[i - 1])
            return float(grid[i - 1] + t * (grid[i] - grid[i - 1]))
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2.0
        cap = _bpsk_capacity_scalar(2.0 * rate * (10.0 ** (mid / 10.0)))
        if cap < rate:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


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
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
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
            K_use = K if K is not None else N // 2
            rate = K_use / N
            info_idx, frozen_idx, _ = ga_construction(
                N, K_use, design_eb_n0_db, rate=rate
            )
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
