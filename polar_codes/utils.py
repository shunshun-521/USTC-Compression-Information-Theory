"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
        "avg_decode_time_ms", "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "eb_n0_db": r["eb_n0_db"],
                "bler": r["bler"],
                "ber": r["ber"],
                "num_errors": r["num_errors"],
                "num_frames": r["num_frames"],
                "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                "avg_iters": "" if r["avg_iters"] is None else r["avg_iters"],
            })


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    I = 1 - (1/sqrt(pi)) ∫_0^∞ exp(-x²) log2(1+exp(-2ρx)) dx, ρ = Eb/N0（线性）
    """
    eb_lin = 10 ** (np.atleast_1d(eb_n0_db_list).astype(np.float64) / 10.0)

    def _symmetric_cap(rho):
        def integrand(x):
            t = -2.0 * rho * x
            if t > 30:
                val = t / np.log(2)
            elif t < -30:
                val = 0.0
            else:
                val = np.log2(1.0 + np.exp(t))
            return np.exp(-x * x) / np.sqrt(np.pi) * val

        val, _ = integrate.quad(integrand, 0.0, np.inf)
        return 1.0 - val

    return np.array([_symmetric_cap(r) for r in eb_lin])


def find_capacity_limit(rate, eb_n0_range=(-20, 10), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if compute_bpsk_capacity(mid, rate)[0] > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线（对数纵轴）"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

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
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
