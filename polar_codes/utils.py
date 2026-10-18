"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
                    r.get("avg_iters", ""),
                ]
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
            row["avg_decode_time"] = float(row["avg_decode_time"])
            ai = row.get("avg_iters", "")
            row["avg_iters"] = float(ai) if ai not in ("", None) else None
            rows.append(row)
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 容量（bits/channel use）"""
    from scipy import integrate

    caps = []
    for eb_db in eb_n0_db_list:
        snr = (10 ** (eb_db / 10.0)) * 2.0 * rate

        def integrand(y):
            t = -2.0 * snr * y
            if t > 50:
                inner = 0.0
            elif t < -50:
                inner = t
            else:
                inner = np.log2(1.0 + np.exp(t))
            return inner * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -20.0, 20.0)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(0.0, 6.0), num_points=600):
    """使 BPSK 容量 C=R 的 Eb/N0（dB）"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if caps[idx] < rate and eb_n0_range[1] < 15:
        grid2 = np.linspace(eb_n0_range[1], 15.0, num_points)
        caps2 = compute_bpsk_capacity(grid2, rate)
        idx2 = np.argmin(np.abs(caps2 - rate))
        return float(grid2[idx2])
    return float(grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER 曲线（PNG + PDF）"""
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
