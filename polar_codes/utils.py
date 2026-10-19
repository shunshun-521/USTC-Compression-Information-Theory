"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV。"""
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
            row["avg_decode_time"] = row["avg_decode_time"] * 1000.0
            if row["avg_iters"] is None:
                row["avg_iters"] = ""
            writer.writerow(row)


def load_results_csv(filepath):
    """从 CSV 加载结果。"""
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"]) / 1000.0
            row["avg_iters"] = float(row["avg_iters"]) if row["avg_iters"] else None
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）。"""
    from scipy import integrate

    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            t = -2.0 * snr * y
            # 数值稳定：log2(1+exp(t)) = log2(1+exp(t)) 分段
            if t > 0:
                log_term = t / np.log(2.0) + np.log1p(np.exp(-t)) / np.log(2.0)
            else:
                log_term = np.log1p(np.exp(t)) / np.log(2.0)
            return log_term * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
        caps.append(1.0 - val / np.sqrt(2.0 * np.pi))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）。"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = int(np.argmin(np.abs(caps - rate)))
    if caps[idx] < rate and idx + 1 < len(grid):
        x0, x1 = grid[idx], grid[idx + 1]
        c0, c1 = caps[idx], caps[idx + 1]
    elif idx > 0:
        x0, x1 = grid[idx - 1], grid[idx]
        c0, c1 = caps[idx - 1], caps[idx]
    else:
        return float(grid[idx])
    if c1 == c0:
        return float(x1)
    return float(x0 + (rate - c0) * (x1 - x0) / (c1 - c0))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


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
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
