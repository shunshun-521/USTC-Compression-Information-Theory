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
    """将仿真结果保存为 CSV。"""
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
                    r.get("avg_iters", ""),
                ]
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return rows


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 信道容量（bits/channel use）数值积分。"""
    from scipy import integrate

    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            return np.log2(1.0 + np.exp(-2.0 * snr * y ** 2)) * np.exp(-y ** 2) / np.sqrt(np.pi)

        cap, _ = integrate.quad(integrand, 0, np.inf, limit=200)
        caps.append(1.0 - cap)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=400):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）。"""
    try:
        from scipy import integrate

        def capacity(eb_db):
            snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

            def integrand(y):
                return (
                    np.log2(1.0 + np.exp(-2.0 * snr * y ** 2))
                    * np.exp(-y ** 2)
                    / np.sqrt(np.pi)
                )

            cap, _ = integrate.quad(integrand, 0, np.inf, limit=200)
            return 1.0 - cap

        lo, hi = eb_n0_range[0], eb_n0_range[1]
        if capacity(lo) < rate:
            return float(lo)
        if capacity(hi) > rate:
            return float(hi)
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if capacity(mid) > rate:
                lo = mid
            else:
                hi = mid
        return float((lo + hi) / 2.0)
    except Exception:
        return 0.19 if abs(rate - 0.5) < 1e-6 else 0.0


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
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(x, y, "o-", label=label)
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
    """保存各码长的信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = N // 2 if K is None else int(K)
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            rate = K_n / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n{info_idx}\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n{frozen_idx}\n")
            f.write("-" * 53 + "\n")
