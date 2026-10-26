"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None


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
                    "avg_iters": r.get("avg_iters", ""),
                }
            )


def load_results_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"])
            row["avg_iters"] = (
                float(row["avg_iters"]) if row.get("avg_iters") not in (None, "") else None
            )
            out.append(row)
        return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 容量 C(Eb/N0)（bits/channel use）。"""
    caps = []
    for eb_db in eb_n0_db_list:
        rho = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            return np.log2(1.0 + np.exp(-2.0 * rho * y)) * np.exp(-(y ** 2) / 2.0)

        if integrate is not None:
            val, _ = integrate.quad(integrand, -np.inf, np.inf)
            cap = 1.0 - val / np.sqrt(2.0 * np.pi)
        else:
            ys = np.linspace(-8, 8, 20000)
            val = np.trapz(integrand(ys), ys)
            cap = 1.0 - val / np.sqrt(2.0 * np.pi)
        caps.append(cap)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=200):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    diff = caps - rate
    cross = np.where(np.diff(np.sign(diff)))[0]
    if cross.size:
        i = int(cross[0])
        t = diff[i] / (diff[i] - diff[i + 1])
        return float(eb_grid[i] + t * (eb_grid[i + 1] - eb_grid[i]))
    return float(eb_grid[int(np.argmin(np.abs(diff)))])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    import matplotlib.pyplot as plt

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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            info, frozen, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={K_n/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n")
            f.write(np.array2string(info, threshold=info.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n")
            f.write(np.array2string(frozen, threshold=frozen.size) + "\n")
            f.write("-" * 53 + "\n")
