"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

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
            avg_iters = r["avg_iters"] if r["avg_iters"] is not None else ""
            writer.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    avg_iters,
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def bpsk_capacity(snr_linear):
    """单点 BPSK 互信息（bits/channel use），Monte Carlo 估计。"""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 20000)
    llr = 2.0 * snr_linear * y
    px0 = 1.0 / (1.0 + np.exp(-llr))
    px1 = 1.0 - px0
    py = 0.5 * px0 + 0.5 * px1
    h_y = -np.mean(np.log2(np.clip(py, 1e-12, 1.0)))
    h_yx = -0.5 * np.mean(np.log2(np.clip(px0, 1e-12, 1.0))) - 0.5 * np.mean(
        np.log2(np.clip(px1, 1e-12, 1.0))
    )
    return h_y - h_yx


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK-AWGN 经典香农容量曲线 C(Eb/N0)（bits/channel use）。"""
    eb = np.asarray(eb_n0_db_list, dtype=np.float64)
    nu = 10.0 ** (eb / 10.0)
    return 0.5 * np.log2(1.0 + 2.0 * rate * nu)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 C(Eb/N0)=R 的 Eb/N0（dB），C 为 BPSK 经典容量界。"""
    nu = (2.0 ** (2.0 * rate) - 1.0) / (2.0 * rate)
    return float(10.0 * np.log10(max(nu, 1e-12)))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

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
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
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
