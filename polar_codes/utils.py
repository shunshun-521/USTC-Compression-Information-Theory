"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time_ms, avg_iters
    """
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
            avg_iters = r.get("avg_iters")
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.4e}",
                    f"{r['ber']:.4e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.4f}",
                    "" if avg_iters is None else f"{avg_iters:.2f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                    "avg_iters": (
                        None
                        if row["avg_iters"] == ""
                        else float(row["avg_iters"])
                    ),
                }
            )
    return results


def _bpsk_mutual_info(eb_n0_db, rate):
    """BPSK-AWGN 互信息（bits/channel use）。"""
    esn0 = rate * (10 ** (eb_n0_db / 10.0))
    var = 1.0 / (2.0 * esn0)

    def p_y_given_x(y, x):
        return np.exp(-((y - x) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)

    def p_y(y):
        return 0.5 * p_y_given_x(y, 1.0) + 0.5 * p_y_given_x(y, -1.0)

    def integrand(y):
        py = p_y(y)
        if py < 1e-300:
            return 0.0
        val = 0.0
        for x in (1.0, -1.0):
            pyx = p_y_given_x(y, x)
            if pyx <= 0.0:
                continue
            ratio = pyx / py
            if ratio <= 0.0:
                continue
            val += 0.25 * pyx * np.log2(ratio)
        return val

    mi, _ = integrate.quad(integrand, -12.0, 12.0, limit=200)
    return mi


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    scalar = np.ndim(eb_n0_db_list) == 0
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = np.array([_bpsk_mutual_info(eb, rate) for eb in eb_n0_db_list])
    return float(caps[0]) if scalar else caps


def find_capacity_limit(rate, eb_n0_range=(-2, 20), num_points=1000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    这是香农限，用于在 BLER 图中标注参考竖线。
    """
    lo, hi = eb_n0_range
    c_lo = compute_bpsk_capacity(lo, rate)
    c_hi = compute_bpsk_capacity(hi, rate)
    eps = 1e-6
    if c_lo >= rate - eps:
        return float(lo)
    if c_hi < rate - eps:
        return float(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if compute_bpsk_capacity(mid, rate) >= rate - eps:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            linewidth=1.2,
            label=f"Capacity limit ({shannon_limit_db:.2f} dB)",
        )

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
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_eff = K if K is not None else N // 2
            rate = K_eff / N
            info_idx, frozen_idx, _ = ga_construction(N, K_eff, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_eff}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120))
            f.write("\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120))
            f.write("\n")
            f.write("-" * 53 + "\n")
