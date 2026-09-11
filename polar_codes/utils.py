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
    """将仿真结果保存为 CSV 文件。"""
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
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
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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


def _snr_linear_from_eb_n0_db(eb_n0_db, rate):
    return 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use）。
    C = 1 - H(Y|X)，其中 H(Y|X) = E[log2(1 + exp(-|L|))], L = 2y/sigma^2。
    """
    from scipy import integrate

    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = _snr_linear_from_eb_n0_db(eb_n0_db, rate)
        sigma = 1.0 / np.sqrt(snr)
        sigma_y = np.sqrt(1.0 + sigma ** 2)

        def integrand(y):
            llr = 2.0 * y / (sigma ** 2)
            hx = np.log2(1.0 + np.exp(-np.abs(llr)))
            return hx * np.exp(-0.5 * (y / sigma_y) ** 2)

        val, _ = integrate.quad(
            integrand,
            -20.0 * sigma_y,
            20.0 * sigma_y,
            limit=200,
        )
        norm = 1.0 / (sigma_y * np.sqrt(2.0 * np.pi))
        capacities.append(1.0 - norm * val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2, label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
