"""工具函数：结果保存、绘图、容量计算"""
import os
import csv
import numpy as np
from scipy import integrate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
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
                "avg_decode_time",
                "avg_iters",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"],
                    r.get("avg_iters") if r.get("avg_iters") is not None else "",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
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
                    "avg_decode_time": float(row["avg_decode_time"]),
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb in eb_n0_db_list:
        eb_lin = 10.0 ** (eb / 10.0)
        es_n0 = rate * eb_lin
        gamma = 2.0 * es_n0

        def integrand(y):
            p = np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)
            t = -gamma * np.abs(y)
            if t > 0:
                log_term = (t + np.log1p(np.exp(-t))) / np.log(2.0)
            else:
                log_term = np.log1p(np.exp(t)) / np.log(2.0)
            return p * (1.0 - log_term)

        val, _ = integrate.quad(integrand, -15.0, 15.0)
        caps.append(val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=2000):
    """找到 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    caps_lo = compute_bpsk_capacity([lo], rate)[0]
    caps_hi = compute_bpsk_capacity([hi], rate)[0]
    if caps_lo < rate:
        lo = -2.0
    if caps_hi > rate:
        hi = 12.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            rate = K_use / N
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
