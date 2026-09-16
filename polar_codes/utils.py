"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

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
                "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
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


def _bpsk_capacity(snr_linear, num_samples=400_000, seed=0):
    """BPSK 离散输入信道容量（bits/channel use），snr = 2R * Eb/N0_linear = 1/sigma^2"""
    sigma = 1.0 / np.sqrt(snr_linear)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_samples)
    x = 1.0 - 2.0 * bits
    y = x + rng.normal(0.0, sigma, size=num_samples)
    llr = 2.0 * y / (sigma ** 2)
    p0 = 1.0 / (1.0 + np.exp(-llr))
    hx = -p0 * np.log2(p0 + 1e-300) - (1.0 - p0) * np.log2(1.0 - p0 + 1e-300)
    return 1.0 - float(np.mean(hx))


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算各 Eb/N0 点对应的 BPSK 信道容量"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        caps.append(_bpsk_capacity(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), tol=0.01):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    for _ in range(40):
        mid = (lo + hi) / 2.0
        cap = _bpsk_capacity(2.0 * rate * (10.0 ** (mid / 10.0)))
        if cap < rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1,
                   label=f"Capacity (R), {shannon_limit_db:.2f} dB")
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
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_n = K if K is not None else N // 2
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
