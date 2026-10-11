"""工具函数：结果保存、绘图、容量计算"""
import numpy as np
import csv
import os
from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use），蒙特卡洛估计互信息。
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = []
    rng = np.random.default_rng(0)
    for eb_n0_db in eb_n0_db_list:
        sigma = 1.0 / np.sqrt(2.0 * rate * (10.0 ** (eb_n0_db / 10.0)))
        x = rng.choice([-1.0, 1.0], size=200000)
        n = rng.normal(0.0, sigma, size=x.shape[0])
        y = x + n
        llr = 2.0 * y / (sigma ** 2)
        px0 = 0.5 * np.exp(-0.5 * ((y - 1.0) / sigma) ** 2)
        px1 = 0.5 * np.exp(-0.5 * ((y + 1.0) / sigma) ** 2)
        py = (px0 + px1) / (sigma * np.sqrt(2.0 * np.pi))
        py = np.maximum(py, 1e-300)
        h_y = -np.mean(np.log2(py))
        h_y_given_x = np.mean(np.log2(1.0 + np.exp(-np.abs(llr))))
        caps.append(max(0.0, h_y - h_y_given_x))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 8), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.where(caps >= rate)[0]
    if len(idx) == 0:
        snr_linear = (2.0 ** (2.0 * rate) - 1.0) / (2.0 * rate)
        return float(10.0 * np.log10(snr_linear))
    if idx[0] == 0:
        return float(grid[0])
    i = idx[0]
    c0, c1 = caps[i - 1], caps[i]
    g0, g1 = grid[i - 1], grid[i]
    if c1 == c0:
        return float(g1)
    return float(g0 + (rate - c0) * (g1 - g0) / (c1 - c0))


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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
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
            f.write(np.array2string(info_idx, threshold=info_idx.size, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
