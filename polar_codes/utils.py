"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
                    "avg_iters": (
                        None
                        if row["avg_iters"] == ""
                        else float(row["avg_iters"])
                    ),
                }
            )
    return results


from scipy import integrate
from scipy.special import logsumexp


def _log2_1_plus_exp(x):
    return logsumexp([0.0, x], axis=0) / np.log(2.0)


def compute_bpsk_capacity(eb_n0_db, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    snr = (10.0 ** (eb_n0_db / 10.0)) * 2.0 * rate

    def integrand(y):
        x = -2.0 * snr * (y ** 2)
        return _log2_1_plus_exp(x) * np.exp(-(y**2) / 2.0)

    val, _ = integrate.quad(integrand, -20.0, 20.0)
    val /= np.sqrt(2.0 * np.pi)
    return 1.0 - val


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range
    c_lo = compute_bpsk_capacity(lo, rate)
    c_hi = compute_bpsk_capacity(hi, rate)
    if c_lo > rate:
        return lo
    if c_hi < rate:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if compute_bpsk_capacity(mid, rate) < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

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
    """保存各码长的信息位/冻结位集合。"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            rate = K_val / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_val}):\n")
            f.write(np.array2string(info_idx, threshold=K_val) + "\n")
            f.write(f"Frozen indices (all {N - K_val}):\n")
            f.write(np.array2string(frozen_idx, threshold=N - K_val) + "\n")
            f.write("-" * 53 + "\n")
