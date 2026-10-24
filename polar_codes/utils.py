"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
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
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if avg_iters is None else f"{avg_iters:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
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


def _bpsk_capacity_per_use(snr_linear):
    """单信道使用容量（snr = 2R * Eb/N0 线性）"""

    def integrand(y):
        z = -2.0 * snr_linear * y
        # log2(1+exp(z)) 的稳定形式
        log_term = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z))) / np.log(2.0)
        return log_term * np.exp(-y ** 2 / 2.0)

    if integrate is not None:
        val, _ = integrate.quad(integrand, -np.inf, np.inf)
        val /= np.sqrt(2.0 * np.pi)
    else:
        y = np.linspace(-8, 8, 20000)
        val = np.sum(integrand(y) * np.exp(-y ** 2 / 2.0)) / np.sqrt(2.0 * np.pi)
        val *= (y[1] - y[0])
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """返回各 Eb/N0 对应的 BPSK 容量（bits/channel use）"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        caps.append(_bpsk_capacity_per_use(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    grid = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(grid, rate)
    diff = caps - rate
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return float(grid[np.argmin(np.abs(diff))])
    i = idx[0]
    x0, x1 = grid[i], grid[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER-Eb/N0 曲线（对数纵轴）"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
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
    """保存各码长的信息位/冻结位集合"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = int(N * rate) if K is None else K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate=K_n / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={K_n/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
