"""工具函数：结果保存、绘图、容量计算"""
import numpy as np
import csv
import os
from scipy import integrate
import matplotlib.pyplot as plt
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
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.4e}",
                    f"{r['ber']:.4e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.4f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.1f}",
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
                    "avg_iters": (
                        float(row["avg_iters"]) if row.get("avg_iters") and row["avg_iters"] else None
                    ),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）"""
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
        capacities.append(_bpsk_mutual_information(snr))
    return capacities


def _bpsk_mutual_information(snr):
    """Es/N0 线性值下的 BPSK-AWGN 互信息（bits/channel use）"""

    def integrand(y):
        p1 = np.exp(-0.5 * (y - np.sqrt(snr)) ** 2) / np.sqrt(2.0 * np.pi)
        p0 = np.exp(-0.5 * (y + np.sqrt(snr)) ** 2) / np.sqrt(2.0 * np.pi)
        py = 0.5 * p1 + 0.5 * p0
        if py <= 0:
            return 0.0
        return -py * np.log2(py)

    hy, _ = integrate.quad(integrand, -30.0, 30.0, limit=200)
    hx = 0.5 * np.log2(2.0 * np.pi * np.e)
    return max(hy - hx, 0.0)


def find_capacity_limit(rate, eb_n0_range=(0, 6), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    for _ in range(60):
        mid = (lo + hi) / 2.0
        snr = 2.0 * rate * (10.0 ** (mid / 10.0))
        cap = _bpsk_mutual_information(snr)
        if cap < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None,
    xlabel="Eb/N0 (dB)", ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
