"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

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
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    f"{r['avg_iters']:.1f}" if r["avg_iters"] is not None else "",
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
                        float(row["avg_iters"]) if row["avg_iters"] else None
                    ),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    通过数值积分计算互信息 I(X;Y)，X ∈ {±1} 等概率。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * 10 ** (eb_n0_db / 10.0)
        sigma = 1.0 / np.sqrt(snr)
        ys = np.linspace(-12.0 * sigma, 12.0 * sigma, 4000)
        dy = ys[1] - ys[0]
        p0 = np.exp(-0.5 * ((ys - 1.0) / sigma) ** 2)
        p1 = np.exp(-0.5 * ((ys + 1.0) / sigma) ** 2)
        norm = np.sqrt(2.0 * np.pi * sigma ** 2)
        p0 /= norm
        p1 /= norm
        py = 0.5 * p0 + 0.5 * p1
        py = np.maximum(py, 1e-300)
        h_y = -np.sum(py * np.log2(py)) * dy
        h_yx = 0.0
        for px in [p0, p1]:
            px = np.maximum(px, 1e-300)
            h_yx += 0.5 * (-np.sum(px * np.log2(px)) * dy)
        capacities.append(h_y - h_yx)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_low, eb_high = eb_n0_range
    eb_vals = np.linspace(eb_low, eb_high, num_points)
    caps = compute_bpsk_capacity(eb_vals, rate)

    for i in range(len(eb_vals) - 1):
        if caps[i] <= rate <= caps[i + 1] or caps[i] >= rate >= caps[i + 1]:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i])
            return eb_vals[i] + t * (eb_vals[i + 1] - eb_vals[i])

    idx = np.argmin(np.abs(caps - rate))
    return eb_vals[idx]


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
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
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
