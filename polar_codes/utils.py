"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            if row.get("avg_iters") is None:
                row["avg_iters"] = ""
            writer.writerow(row)


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"])
            ai = row.get("avg_iters", "")
            row["avg_iters"] = float(ai) if ai not in ("", None) else None
            results.append(row)
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    from scipy import integrate

    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))

        def integrand(y):
            t = -2.0 * snr * y
            # 数值稳定的 log2(1 + exp(t))
            log1pexp = np.maximum(t, 0.0) + np.log1p(np.exp(-np.abs(t)))
            return (log1pexp / np.log(2.0)) * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=800):
    """
    找到使 BPSK 信道互信息等于码率 R 的 Eb/N0（dB）。
    采用 ±1 等概 BPSK-AWGN 的数值互信息积分。
    """
    from scipy import integrate, optimize

    def mutual_info(eb_db):
        gamma = 10 ** (eb_db / 10.0) * 2.0 * rate

        def integrand(y):
            a = np.exp(-0.5 * (y - np.sqrt(gamma)) ** 2)
            b = np.exp(-0.5 * (y + np.sqrt(gamma)) ** 2)
            s = a + b
            s = np.maximum(s, 1e-300)
            p1 = a / s
            p0 = 1.0 - p1
            h = 0.0
            for p in (p0, p1):
                p = np.clip(p, 1e-15, 1.0 - 1e-15)
                h -= p * np.log2(p)
            return h * s / np.sqrt(2.0 * np.pi)

        hy, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
        return 1.0 - hy

    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([mutual_info(eb) for eb in eb_grid])
    for i in range(len(eb_grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            return float(
                optimize.brentq(lambda eb: mutual_info(eb) - rate, eb_grid[i], eb_grid[i + 1])
            )
    return float(eb_grid[np.argmin(np.abs(caps - rate))])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    if plt is None:
        return
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


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
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
            f.write(np.array2string(info_idx, threshold=info_idx.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size) + "\n")
            f.write("-" * 53 + "\n")
