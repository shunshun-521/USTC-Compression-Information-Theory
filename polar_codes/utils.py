"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
                    "avg_iters": None if row.get("avg_iters", "") == "" else float(row["avg_iters"]),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    from scipy import integrate

    caps = []
    for eb in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb / 10))

        def integrand(y):
            val = -snr * y * y
            if val < -50:
                inner = 0.0
            elif val > 50:
                inner = np.exp(val)
            else:
                inner = np.log2(1.0 + np.exp(val))
            return inner * np.exp(-y * y) / np.sqrt(np.pi)

        cap, _ = integrate.quad(integrand, 0, 12, limit=200)
        caps.append(1.0 - cap)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 15), num_points=800):
    """BPSK 上使信道容量近似等于码率 R 的 Eb/N0（dB），用于图中参考竖线。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx > 0 and (caps[idx - 1] - rate) * (caps[idx] - rate) <= 0:
        x0, x1 = eb_grid[idx - 1], eb_grid[idx]
        y0, y1 = caps[idx - 1], caps[idx]
        return float(x0 + (rate - y0) * (x1 - x0) / (y1 - y0))
    # 闭式近似（BPSK 香农限）
    lin = (2 ** (2 * rate) - 1) / (2 * rate)
    return float(10 * np.log10(max(lin, 1e-12)))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon ≈ {shannon_limit_db:.2f} dB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
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
            K_use = K if K is not None else N // 2
            rate = K_use / N
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n{np.array2string(info_idx, threshold=N)}\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n{np.array2string(frozen_idx, threshold=N)}\n")
            f.write("-" * 53 + "\n")
