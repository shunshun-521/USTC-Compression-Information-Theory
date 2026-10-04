"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真 CSV"""
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
    """加载 CSV 结果"""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
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
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use），与 channel.py 的 SNR 定义一致"""
    y = np.linspace(-12.0, 12.0, 24001)
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))
        sigma = 1.0 / np.sqrt(snr)
        py0 = np.exp(-((y - 1.0) ** 2) / (2.0 * sigma ** 2))
        py1 = np.exp(-((y + 1.0) ** 2) / (2.0 * sigma ** 2))
        norm = np.sqrt(2.0 * np.pi * sigma ** 2)
        py0 /= norm
        py1 /= norm
        p_y = 0.5 * (py0 + py1)
        integrand = np.zeros_like(y)
        for py_cond, px in ((py0, 0.5), (py1, 0.5)):
            joint = px * py_cond
            ratio = np.divide(
                joint,
                px * p_y,
                out=np.ones_like(y),
                where=(p_y > 1e-300),
            )
            integrand += np.where(joint > 0, joint * np.log2(ratio), 0.0)
        caps.append(float(np.trapezoid(integrand, y)))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    if np.all(caps < rate):
        return float(eb_n0_range[1])
    if np.all(caps > rate):
        return float(eb_n0_range[0])
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        return float(eb_grid[idx])
    e0, e1 = eb_grid[idx - 1], eb_grid[idx + 1]
    c0, c1 = caps[idx - 1], caps[idx + 1]
    if c1 == c0:
        return float(eb_grid[idx])
    return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER 曲线（PNG + PDF）"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            rate = k_val / N
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
