"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

from construction import ga_construction


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
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
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
                ]
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate=None):
    """BPSK-AWGN 信道容量（bits / channel use）"""
    from scipy import integrate
    from scipy.special import logsumexp

    caps = []
    for eb_n0_db in eb_n0_db_list:
        gamma = 10.0 ** (eb_n0_db / 10.0)

        def integrand(t):
            z = -2.0 * gamma - 2.0 * t * np.sqrt(2.0 * gamma)
            log_term = logsumexp(np.array([0.0, z])) / np.log(2.0)
            return np.exp(-(t ** 2)) * log_term / np.sqrt(np.pi)

        val, _ = integrate.quad(integrand, -20.0, 20.0)
        caps.append(val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=1000):
    from scipy.optimize import brentq

    def diff(eb_db):
        return compute_bpsk_capacity([eb_db])[0] - rate

    lo, hi = eb_n0_range
    while diff(lo) < 0 and lo > -10:
        lo -= 1.0
    while diff(hi) > 0 and hi < 20:
        hi += 1.0
    try:
        return float(brentq(diff, lo, hi))
    except ValueError:
        grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
        caps = compute_bpsk_capacity(grid)
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
        ys = [max(r["bler"], 1e-6) for r in results]
        plt.semilogy(xs, ys, marker="o", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
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


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = int(K) if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate=k_val / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={k_val / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
