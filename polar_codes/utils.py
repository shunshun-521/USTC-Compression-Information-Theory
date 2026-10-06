"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = dict(r)
            row["avg_decode_time"] = row["avg_decode_time"] * 1000.0
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
            row["avg_iters"] = float(row["avg_iters"]) if row.get("avg_iters") else None
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    BPSK 离散输入信道容量（bits/channel use）。
    C = 1 - (1/sqrt(pi)) ∫ log2(1+exp(-4*gamma*x^2)) exp(-x^2) dx, gamma = Eb/N0（线性）
    """
    from scipy import integrate

    caps = []
    for eb_db in eb_n0_db_list:
        gamma = 10.0 ** (eb_db / 10.0)

        def integrand(x):
            return (
                np.log2(1.0 + np.exp(-4.0 * gamma * x * x))
                * np.exp(-x * x)
                / np.sqrt(np.pi)
            )

        val, _ = integrate.quad(integrand, -15.0, 15.0, limit=200)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=80):
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    for _ in range(40):
        mid = (lo + hi) / 2.0
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=2, markersize=5)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon ≈ {shannon_limit_db:.2f} dB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    lines = []
    for N in N_list:
        K_use = K if K is not None else N // 2
        rate = K_use / N
        info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate)
        lines.append("=" * 53)
        lines.append(f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}")
        lines.append("=" * 53)
        lines.append(f"Info indices (all {len(info_idx)}):")
        lines.append(np.array2string(info_idx, separator=" "))
        lines.append(f"Frozen indices (all {len(frozen_idx)}):")
        lines.append(np.array2string(frozen_idx, separator=" "))
        lines.append("-" * 53)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
