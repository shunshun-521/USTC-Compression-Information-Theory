"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    cols = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in cols}
            row["avg_decode_time"] = row["avg_decode_time"] * 1000
            w.writerow(row)


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"]) / 1000.0
            row["avg_iters"] = (
                float(row["avg_iters"]) if row.get("avg_iters") not in ("", None) else None
            )
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db, rate):
    """BPSK-AWGN 信道容量（bits/channel use），s = 2R·10^(Eb/N0/10)。"""
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

    def integrand(y):
        return np.log2(1.0 + np.exp(-snr * y * y)) * np.exp(-0.5 * y * y)

    val, _ = integrate.quad(integrand, 0.0, 50.0, limit=200)
    return 1.0 - val / np.sqrt(2.0 * np.pi)


def find_capacity_limit(rate, eb_n0_range=(-6, 8), num_points=400):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = [compute_bpsk_capacity(eb, rate) for eb in grid]
    for i in range(1, len(grid)):
        if (caps[i - 1] - rate) * (caps[i] - rate) <= 0:
            t = (rate - caps[i - 1]) / (caps[i] - caps[i - 1] + 1e-12)
            return float(grid[i - 1] + t * (grid[i] - grid[i - 1]))
    return float(grid[0] if caps[0] < rate else grid[-1])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    lines = []
    for N in N_list:
        k = K if K is not None else N // 2
        info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
        lines.append("=" * 53)
        lines.append(
            f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={k/N:.4f}"
        )
        lines.append("=" * 53)
        lines.append(f"Info indices (all {len(info_idx)}):")
        lines.append(np.array2string(info_idx, threshold=N))
        lines.append(f"Frozen indices (all {len(frozen_idx)}):")
        lines.append(np.array2string(frozen_idx, threshold=N))
        lines.append("-" * 53)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
