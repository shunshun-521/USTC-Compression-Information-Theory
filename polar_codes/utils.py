"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


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
            writer.writerow(
                {
                    "eb_n0_db": f"{r['eb_n0_db']:.2f}",
                    "bler": f"{r['bler']:.6e}",
                    "ber": f"{r['ber']:.6e}",
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": f"{r['avg_decode_time'] * 1000:.6f}",
                    "avg_iters": (
                        f"{r['avg_iters']:.2f}" if r["avg_iters"] is not None else ""
                    ),
                }
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
                        float(row["avg_iters"]) if row["avg_iters"].strip() else None
                    ),
                }
            )
    return results


def _bpsk_mi_single(ebno_linear):
    """BPSK BI-AWGN 互信息（bits/channel use），Y = sqrt(2*Eb/N0)*X + Z"""
    a = np.sqrt(2.0 * ebno_linear)
    ys = np.linspace(-15.0, 15.0, 60001)
    p0 = np.exp(-0.5 * (ys - a) ** 2)
    p1 = np.exp(-0.5 * (ys + a) ** 2)
    py = 0.5 * (p0 + p1)
    py /= np.trapezoid(py, ys)
    hy = -np.trapezoid(py * np.log2(py + 1e-300), ys)
    hyx = 0.0
    for sign in (1, -1):
        p_yx = np.exp(-0.5 * (ys - sign * a) ** 2)
        p_yx /= np.trapezoid(p_yx, ys)
        hyx += 0.5 * (-np.trapezoid(p_yx * np.log2(p_yx + 1e-300), ys))
    return hy - hyx


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK-AWGN 信道互信息（bits/channel use）"""
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = []
    for eb_n0_db in eb_n0_db_list:
        ebno_lin = 10.0 ** (eb_n0_db / 10.0)
        caps.append(_bpsk_mi_single(2.0 * rate * ebno_lin))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道互信息等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range

    def objective(eb_db):
        ebno_lin = 10.0 ** (eb_db / 10.0)
        return _bpsk_mi_single(2.0 * rate * ebno_lin) - rate

    if objective(lo) * objective(hi) > 0:
        grid = np.linspace(lo, hi, num_points)
        vals = [objective(x) for x in grid]
        idx = int(np.argmin(np.abs(vals)))
        return float(grid[idx])

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if objective(mid) > 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


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


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={k_val / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ") + "\n")
            f.write("-" * 53 + "\n")
