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
    """保存仿真结果为 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eb_n0_db",
                "bler",
                "ber",
                "num_errors",
                "num_frames",
                "avg_decode_time",
                "avg_iters",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"],
                    r.get("avg_iters") if r.get("avg_iters") is not None else "",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    out = []
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
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate, num_samples=80000, seed=0):
    """
    BPSK-AWGN 互信息（bits/channel use）的蒙特卡洛估计。
    与信道实现一致：sigma = 1/sqrt(2*R*Eb/N0_lin)。
    """
    rng = np.random.default_rng(seed)
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        sigma = 1.0 / np.sqrt(snr)
        x = rng.choice(np.array([-1.0, 1.0]), size=num_samples)
        y = x + rng.normal(0.0, sigma, size=num_samples)
        llr = 2.0 * y / (sigma ** 2)
        p0 = 1.0 / (1.0 + np.exp(-llr))
        p0 = np.clip(p0, 1e-12, 1.0 - 1e-12)
        h = -p0 * np.log2(p0) - (1.0 - p0) * np.log2(1.0 - p0)
        caps.append(1.0 - float(np.mean(h)))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(grid) - 1:
        return float(grid[idx])
    # 线性插值
    for i in range(len(grid) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0:
            t = (rate - c0) / (c1 - c0 + 1e-15)
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    return float(grid[idx])


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER 曲线（PNG + PDF）"""
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
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
            info, frozen, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n")
            f.write(np.array2string(info, threshold=info.size) + "\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n")
            f.write(np.array2string(frozen, threshold=frozen.size) + "\n")
            f.write("-" * 53 + "\n")
