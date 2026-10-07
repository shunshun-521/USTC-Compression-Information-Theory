"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from construction import ga_construction


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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    BPSK 离散输入 AWGN 信道容量（bits/channel use）。
    采用标准对称二进制输入互信息数值积分。
    """
    caps = []
    for eb in eb_n0_db_list:
        esn0 = rate * (10.0 ** (eb / 10.0))
        sigma = 1.0 / np.sqrt(2.0 * esn0) if esn0 > 0 else 1e6

        def integrand(y):
            p0 = np.exp(-((y - 1.0) ** 2) / (2.0 * sigma ** 2))
            p1 = np.exp(-((y + 1.0) ** 2) / (2.0 * sigma ** 2))
            py = (p0 + p1) / (2.0 * sigma * np.sqrt(2.0 * np.pi))
            px0, px1 = 0.5, 0.5
            py0 = p0 / (sigma * np.sqrt(2.0 * np.pi))
            py1 = p1 / (sigma * np.sqrt(2.0 * np.pi))
            py_given_x = px0 * py0 + px1 * py1
            if py_given_x <= 0:
                return 0.0
            term = 0.0
            for px, pyx in ((px0, py0), (px1, py1)):
                if pyx > 0:
                    term += px * pyx / py_given_x * np.log2(px * pyx / py_given_x)
            return py_given_x * term

        val, _ = integrate.quad(integrand, -30.0 * sigma - 1.0, 30.0 * sigma + 1.0, limit=200)
        caps.append(max(0.0, min(1.0, -val)))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(grid) - 1:
        return float(grid[idx])
    e0, e1 = grid[idx - 1], grid[idx]
    c0, c1 = caps[idx - 1], caps[idx]
    if c1 == c0:
        return float(e1)
    return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
