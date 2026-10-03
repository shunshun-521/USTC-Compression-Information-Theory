"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV"""
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
            writer.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    "" if r.get("avg_iters") is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    out = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
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
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64)
    caps = []
    y = np.linspace(-10, 10, 4001)
    dy = y[1] - y[0]
    pdf0 = np.exp(-0.5 * y ** 2) / np.sqrt(2 * np.pi)
    for eb in eb_n0_db_list:
        snr_lin = 2.0 * rate * (10.0 ** (eb / 10.0))
        integrand = np.log2(1.0 + np.exp(-snr_lin * y ** 2)) * pdf0
        caps.append(1.0 - np.sum(integrand) * dy)
    return np.asarray(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=2000):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    caps_lo = compute_bpsk_capacity([lo], rate)[0]
    caps_hi = compute_bpsk_capacity([hi], rate)[0]
    if caps_lo < rate:
        lo = -2.0
    if caps_hi > rate:
        hi = 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER 曲线（PNG + PDF）"""
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    if K is None:
        K = lambda N: N // 2
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k = K(N) if callable(K) else K
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
            rate = k / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {k}):\n{np.array2string(info_idx, threshold=N)}\n")
            f.write(
                f"Frozen indices (all {N - k}):\n{np.array2string(frozen_idx, threshold=N)}\n"
            )
            f.write("-" * 53 + "\n")
