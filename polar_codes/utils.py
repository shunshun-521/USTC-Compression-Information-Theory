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
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
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
                    "" if r.get("avg_iters") is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="") as f:
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate=None):
    """
    BPSK-AWGN 信道容量（bits/channel use），与码率 R 比较用于香农限。
    rate 参数保留接口兼容性，不参与积分。
    """
    from scipy import integrate

    caps = []
    for eb_n0_db in eb_n0_db_list:
        eb_lin = 10.0 ** (eb_n0_db / 10.0)

        def integrand(y):
            z = -4.0 * eb_lin * (y ** 2)
            if z < -40.0:
                term = 0.0
            elif z > 40.0:
                term = z / np.log(2.0)
            else:
                term = np.log2(1.0 + np.exp(z))
            return term * np.exp(-y * y) / np.sqrt(np.pi)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    from scipy import optimize

    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid)

    def f(eb):
        return compute_bpsk_capacity([eb])[0] - rate

    idx = np.argmin(np.abs(caps - rate))
    x0 = grid[idx]
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return float(grid[np.argmin(np.abs(caps - rate))])
    root = optimize.brentq(f, lo, hi)
    return float(root)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
            info_idx, frozen_idx, _ = ga_construction(N, K_loc, design_eb_n0_db)
            rate = K_loc / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_loc}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_loc}):\n")
            f.write(np.array2string(info_idx, threshold=K_loc) + "\n")
            f.write(f"Frozen indices (all {N - K_loc}):\n")
            f.write(np.array2string(frozen_idx, threshold=N - K_loc) + "\n")
            f.write("-" * 53 + "\n")
