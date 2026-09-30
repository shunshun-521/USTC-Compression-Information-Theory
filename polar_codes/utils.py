"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fields = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
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
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入 AWGN 信道容量（bits/channel use）"""
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        s = np.sqrt(snr)

        def integrand(y):
            p = np.exp(-((y - s) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
            q = np.exp(-((y + s) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
            denom = p + q
            num = p
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(denom > 0, num / denom, 0.5)
                inner = 1.0 + np.exp(-2.0 * s * y) * (term / (1.0 - term + 1e-30) - 1.0)
            # 标准形式 C = 1 - E[log2(1+exp(-2sy))] 对 BPSK
            ll = np.log2(1.0 + np.exp(-2.0 * s * y))
            return ll * np.exp(-(y ** 2) / 2.0) / np.sqrt(2.0 * np.pi)

        cap, _ = integrate.quad(integrand, -20, 20, limit=200)
        caps.append(max(0.0, 1.0 - cap))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=800):
    """使 BPSK 容量等于 R 的 Eb/N0（dB）"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    for i in range(len(grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i] + 1e-30)
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_half, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K = N // 2 if K_half is None else K_half
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db)
            rate = K / N
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n{np.array2string(info_idx, threshold=200)}\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n{np.array2string(frozen_idx, threshold=200)}\n")
            f.write("-" * 53 + "\n")
