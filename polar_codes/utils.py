"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
from scipy import integrate

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
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    "" if r.get("avg_iters") is None else r["avg_iters"],
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        def integrand(y):
            t = -2.0 * snr * y
            log_term = np.log2(1.0 + np.exp(np.clip(t, -50, 50)))
            return log_term * np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)
        c = 1.0 - integrate.quad(integrand, -10, 10, limit=200)[0]
        caps.append(c)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    return float(eb_grid[idx])


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
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        plt.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_loc = N // 2
            else:
                K_loc = K
            info, frozen, _ = ga_construction(N, K_loc, design_eb_n0_db)
            rate = K_loc / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_loc}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n")
            f.write(np.array2string(info, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n")
            f.write(np.array2string(frozen, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
