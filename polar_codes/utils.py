"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            avg_iters = r.get("avg_iters")
            writer.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                "" if avg_iters is None else f"{avg_iters:.4f}",
            ])


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
            })
    return results


def bpsk_capacity(eb_n0_db, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
    s = np.sqrt(snr)

    def integrand(y):
        return np.log2(1.0 + np.exp(-2.0 * s * y)) * np.exp(-0.5 * y ** 2) / np.sqrt(2 * np.pi)

    from scipy import integrate
    cap, _ = integrate.quad(integrand, -20, 20)
    return 1.0 - cap


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算一组 Eb/N0 下的 BPSK 容量"""
    return np.array([bpsk_capacity(eb, rate) for eb in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    eb_grid = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        for _ in range(50):
            mid = (lo + hi) / 2
            if bpsk_capacity(mid, rate) > rate:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    return float(eb_grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线（PNG + PDF）"""
    if not HAS_MPL:
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
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                k = N // 2
            else:
                k = K
            rate = k / N
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {k}):\n")
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - k}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
