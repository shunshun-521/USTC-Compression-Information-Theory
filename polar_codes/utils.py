"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            writer.writerow([
                r["eb_n0_db"], r["bler"], r["ber"], r["num_errors"], r["num_frames"],
                r["avg_decode_time"] * 1000,
                "" if r["avg_iters"] is None else r["avg_iters"],
            ])


def load_results_csv(filepath):
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000,
                "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb_n0_db / 10.0))

        def integrand(y):
            t = -2 * snr * y
            log_term = np.log1p(np.exp(np.clip(t, -700, 700))) / np.log(2)
            return log_term * np.exp(-y ** 2 / 2)

        val, _ = integrate.quad(integrand, -20, 20)
        caps.append(1 - val / np.sqrt(2 * np.pi))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=2000):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.searchsorted(caps, rate)
    idx = min(max(idx, 1), len(eb_grid) - 1)
    x0, x1 = eb_grid[idx - 1], eb_grid[idx]
    y0, y1 = caps[idx - 1], caps[idx]
    if y1 == y0:
        return float(x0)
    return float(x0 + (rate - y0) * (x1 - x0) / (y1 - y0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(x, y, "o-", label=label, linewidth=2, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Capacity limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.savefig(save_path.replace(".png", ".pdf"))
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, separator=" ") + "\n")
            f.write("-" * 53 + "\n")
