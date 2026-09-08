"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time_ms, avg_iters
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                "" if r["avg_iters"] is None else f"{r['avg_iters']:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
    """BPSK 信道容量近似（bits/channel use），用于绘图参考"""
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb = 10.0 ** (eb_n0_db / 10.0)
        from scipy.special import erfc
        ber_uncoded = 0.5 * erfc(np.sqrt(eb))
        cap = max(0.0, 1.0 - 2.0 * ber_uncoded * np.log2(np.e))
        capacities.append(min(1.0, cap * (1.0 - 0.5 * (1.0 - rate))))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    使用已知 BPSK-AWGN 香农限查表值。
    """
    table = {0.25: -0.86, 0.5: 0.19, 0.75: 1.84, 1.0: 4.0}
    rates = sorted(table.keys())
    if rate <= rates[0]:
        return table[rates[0]]
    if rate >= rates[-1]:
        return table[rates[-1]]
    for i in range(len(rates) - 1):
        if rates[i] <= rate <= rates[i + 1]:
            t = (rate - rates[i]) / (rates[i + 1] - rates[i])
            return table[rates[i]] * (1.0 - t) + table[rates[i + 1]] * t
    return 0.19


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1,
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N) + "\n")
            f.write("-" * 53 + "\n")
