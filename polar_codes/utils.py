"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
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
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return out


def _bpsk_mi_bits_per_channel(eb_n0_db, rate, rng=None):
    """BPSK-AWGN 互信息（bits/信道使用），Monte Carlo"""
    if rng is None:
        rng = np.random.default_rng(0)
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    sigma = 1.0 / np.sqrt(snr)
    n = 120000
    x = rng.choice(np.array([-1.0, 1.0]), size=n)
    y = x + rng.normal(0.0, sigma, size=n)
    llr = 2.0 * y / (sigma ** 2)
    return float(1.0 - np.mean(np.log2(1.0 + np.exp(-np.abs(llr)))))


def compute_bpsk_capacity(eb_n0_db_list, rate):
    rng = np.random.default_rng(1)
    return np.array([_bpsk_mi_bits_per_channel(eb, rate, rng) for eb in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-2, 12)):
    lo, hi = eb_n0_range
    rng = np.random.default_rng(2)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _bpsk_mi_bits_per_channel(mid, rate, rng) > rate:
            hi = mid
        else:
            lo = mid
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
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={k/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
