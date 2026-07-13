"""
工具函数：结果保存、绘图、容量计算
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 信道互信息（bits/channel use）。
    信号 x ∈ {+1,-1}，噪声 n ~ N(0, σ²)，σ² = 1/(2R·Eb/N0)。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb_lin = 10 ** (eb_n0_db / 10.0)
        if eb_lin <= 0:
            capacities.append(0.0)
            continue
        sigma2 = 1.0 / (2.0 * rate * eb_lin)
        sigma = np.sqrt(sigma2)

        y = np.linspace(-8.0, 8.0, 40000)
        pdf = np.exp(-0.5 * (y ** 2) / sigma2) / (sigma * np.sqrt(2.0 * np.pi))
        p0 = pdf  # placeholder for loop below
        del p0

        p_y = 0.5 * (
            np.exp(-0.5 * ((y - 1.0) ** 2) / sigma2)
            + np.exp(-0.5 * ((y + 1.0) ** 2) / sigma2)
        ) / (sigma * np.sqrt(2.0 * np.pi))

        eps = 1e-300
        h_y = -np.trapezoid(np.where(p_y > eps, p_y * np.log2(p_y), 0.0), y)

        p_y0 = np.exp(-0.5 * ((y - 1.0) ** 2) / sigma2) / (sigma * np.sqrt(2.0 * np.pi))
        p_y1 = np.exp(-0.5 * ((y + 1.0) ** 2) / sigma2) / (sigma * np.sqrt(2.0 * np.pi))
        h_y_given_x = 0.5 * (
            -np.trapezoid(np.where(p_y0 > eps, p_y0 * np.log2(p_y0), 0.0), y)
            + -np.trapezoid(np.where(p_y1 > eps, p_y1 * np.log2(p_y1), 0.0), y)
        )
        capacities.append(h_y - h_y_given_x)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    diff = caps - rate
    # 找符号变化区间
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        lo, hi = eb_grid[i], eb_grid[i + 1]
        for _ in range(50):
            mid = (lo + hi) / 2.0
            cap = compute_bpsk_capacity([mid], rate)[0]
            if cap < rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    return eb_grid[np.argmin(np.abs(diff))]


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

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
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位/冻结位集合保存到文本文件"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(
                N, K_val, design_eb_n0_db, rate=rate
            )
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
