"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time": r["avg_decode_time"],
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"])
            row["avg_iters"] = (
                float(row["avg_iters"]) if row.get("avg_iters", "") != "" else None
            )
            out.append(row)
    return out


def _log2_one_plus_exp(neg_x):
    if neg_x > 50.0:
        return -neg_x / np.log(2.0)
    return np.log2(1.0 + np.exp(-neg_x))


def _bpsk_capacity_per_snr(snr_linear):
    def integrand(y):
        t = 2.0 * snr_linear * y
        return _log2_one_plus_exp(t) * np.exp(-(y ** 2) / 2.0)

    val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
    return 1.0 - val / np.sqrt(2.0 * np.pi)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb_db / 10.0))
        caps.append(_bpsk_capacity_per_snr(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    return float(grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(xs, ys, marker="o", linewidth=1.5, label=label)

    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_eff = K if K is not None else N // 2
            rate = K_eff / N
            info_idx, frozen_idx, _ = ga_construction(N, K_eff, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_eff}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
