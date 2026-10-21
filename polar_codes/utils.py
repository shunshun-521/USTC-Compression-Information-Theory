"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
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
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def _bpsk_capacity_per_eb_n0(eb_n0_db, rate, num_samples=80000):
    """BPSK-AWGN 信道容量（Monte Carlo 估计互信息）"""
    snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
    sigma = 1.0 / np.sqrt(snr)
    rng = np.random.default_rng(int(abs(eb_n0_db * 1000)) % (2**32 - 1))
    x = rng.choice([-1.0, 1.0], num_samples)
    y = x + sigma * rng.standard_normal(num_samples)
    llr = 2.0 * y / (sigma ** 2)
    p0 = 1.0 / (1.0 + np.exp(-llr))
    p1 = 1.0 - p0
    h = -(p0 * np.log2(p0 + 1e-300) + p1 * np.log2(p1 + 1e-300))
    return 1.0 - float(np.mean(h))


def compute_bpsk_capacity(eb_n0_db_list, rate):
    return np.array([_bpsk_capacity_per_eb_n0(e, rate) for e in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    for i in range(len(eb_grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i] + 1e-12)
            return float(eb_grid[i] + t * (eb_grid[i + 1] - eb_grid[i]))
    idx = int(np.argmin(np.abs(caps - rate)))
    return float(eb_grid[idx])


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
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate=K_use / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={K_use/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_use}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - K_use}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
