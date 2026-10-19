"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV"""
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time_ms"]) / 1000.0
            row["avg_iters"] = (
                float(row["avg_iters"]) if row["avg_iters"] not in ("", None) else None
            )
            rows.append(row)
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK-AWGN 离散输入互信息（bits/channel use），与 channel.py 调制一致。"""
    from channel import eb_n0_to_sigma

    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64)
    caps = []
    y = np.linspace(-6.0, 6.0, 12000)
    dy = y[1] - y[0]
    for eb_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(float(eb_db), rate)
        var = sigma ** 2
        p_y = 0.5 * np.exp(-((y - 1.0) ** 2) / (2.0 * var)) + 0.5 * np.exp(
            -((y + 1.0) ** 2) / (2.0 * var)
        )
        p_y /= np.sqrt(2.0 * np.pi * var)
        p0 = np.exp(-((y - 1.0) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
        p1 = np.exp(-((y + 1.0) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
        p0y = np.clip(0.5 * p0 / np.maximum(p_y, 1e-300), 1e-300, 1.0)
        p1y = np.clip(0.5 * p1 / np.maximum(p_y, 1e-300), 1e-300, 1.0)
        h = -(p0y * np.log2(p0y) + p1y * np.log2(p1y))
        cap = np.sum(p_y * h) * dy
        caps.append(cap)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    diff = caps - rate
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return float(eb_grid[np.argmin(np.abs(diff))])
    i = idx[0]
    x0, x1 = eb_grid[i], eb_grid[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


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
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label, markersize=4)
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
    """保存信息位/冻结位集合"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_use = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={K_use / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
