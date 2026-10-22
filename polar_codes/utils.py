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
    """保存仿真结果为 CSV。"""
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
    """从 CSV 加载仿真结果。"""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
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
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）。"""
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    cap = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        s = np.sqrt(snr)
        y = np.linspace(-12.0, 12.0, 24001)
        dy = y[1] - y[0]
        t = -2.0 * s * y
        log_term = np.empty_like(y)
        pos = t > 0
        log_term[pos] = t[pos] / np.log(2.0) + np.log2(1.0 + np.exp(-t[pos]))
        log_term[~pos] = np.log2(1.0 + np.exp(t[~pos]))
        integrand = log_term * np.exp(-0.5 * y * y)
        val = np.sum(integrand) * dy / np.sqrt(2.0 * np.pi)
        cap.append(1.0 - val)
    return np.array(cap)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    """找到 BPSK 容量等于码率 R 的 Eb/N0（dB）。"""
    from scipy import optimize

    def capacity_minus_rate(eb_db):
        gamma_b = 10 ** (eb_db / 10.0)
        gamma_s = rate * gamma_b
        s = np.sqrt(2.0 * gamma_s)
        y = np.linspace(0.0, 14.0, 12001)
        dy = y[1] - y[0]
        t = -2.0 * s * y
        log_term = np.empty_like(y)
        pos = t > 0
        log_term[pos] = t[pos] / np.log(2.0) + np.log2(1.0 + np.exp(-t[pos]))
        log_term[~pos] = np.log2(1.0 + np.exp(t[~pos]))
        integrand = log_term * np.exp(-0.5 * y * y)
        val = 2.0 * np.sum(integrand) * dy / np.sqrt(2.0 * np.pi)
        cap = 1.0 - val
        return cap - rate

    try:
        return float(optimize.brentq(capacity_minus_rate, eb_n0_range[0], eb_n0_range[1]))
    except ValueError:
        # 常用码率参考值（BPSK，近似）
        if abs(rate - 0.5) < 1e-6:
            return 0.19
        eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
        caps = [capacity_minus_rate(eb) + rate for eb in eb_grid]
        idx = int(np.argmin(np.abs(np.array(caps) - rate)))
        return float(eb_grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线（PNG + PDF）。"""
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
