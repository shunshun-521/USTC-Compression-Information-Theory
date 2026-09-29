"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

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
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.3f}",
                ]
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
                    "avg_iters": None if row["avg_iters"] == "" else float(row["avg_iters"]),
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    eb = np.asarray(eb_n0_db_list, dtype=np.float64)
    snr = 2.0 * rate * (10.0 ** (eb / 10.0))
    caps = []
    for s in snr:
        y = np.linspace(-10.0 / np.sqrt(s), 10.0 / np.sqrt(s), 20001)
        pdf = np.exp(-0.5 * y ** 2) / np.sqrt(2.0 * np.pi)
        integrand = np.log2(1.0 + np.exp(-2.0 * s * y ** 2))
        caps.append(1.0 - np.trapezoid(integrand * pdf, y))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    return float(eb_grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, marker="o", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon (R) ≈ {shannon_limit_db:.2f} dB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={K_use / N:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
