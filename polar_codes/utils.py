"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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
            avg_iters = r.get("avg_iters")
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if avg_iters is None else f"{avg_iters:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
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
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters", "").strip()
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    BPSK 离散输入信道容量（bits/channel use）。
    C = 1 - (2/sqrt(2pi)) * int_0^inf log2(1+exp(-2*snr*y^2)) exp(-y^2/2) dy
    snr = 2R * 10^{Eb/N0/10}
    """
    from scipy import integrate

    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(z):
            x = -2.0 * snr * z * z
            lx = np.where(
                x > 0,
                x / np.log(2) + np.log1p(np.exp(-x)) / np.log(2),
                np.log1p(np.exp(x)) / np.log(2),
            )
            return lx * np.exp(-0.5 * z * z)

        val, _ = integrate.quad(integrand, 0.0, 20.0)
        val = 2.0 * val / np.sqrt(2.0 * np.pi)
        capacities.append(1.0 - val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.searchsorted(caps, rate)
    if idx == 0:
        return float(eb_grid[0])
    if idx >= len(eb_grid):
        return float(eb_grid[-1])
    c0, c1 = caps[idx - 1], caps[idx]
    e0, e1 = eb_grid[idx - 1], eb_grid[idx]
    if abs(c1 - c0) < 1e-12:
        return float(e0)
    return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线（semilogy）"""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
