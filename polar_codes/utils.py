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
    with open(filepath, "w", newline="") as f:
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
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
    rows = []
    with open(filepath, newline="") as f:
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


def _bpsk_mutual_info(snr_lin):
    """BPSK ±1 AWGN 互信息（bits/channel use），snr_lin = 1/sigma^2"""
    from scipy import integrate

    sigma = 1.0 / np.sqrt(snr_lin)

    def part(xsign):
        def integrand(y):
            norm = np.sqrt(2.0 * np.pi) * sigma
            pyx = np.exp(-((y - xsign) ** 2) / (2.0 * sigma ** 2)) / norm
            py = 0.5 * (
                np.exp(-((y - 1) ** 2) / (2.0 * sigma ** 2))
                + np.exp(-((y + 1) ** 2) / (2.0 * sigma ** 2))
            ) / norm
            if py <= 0.0:
                return 0.0
            return pyx * np.log2(pyx / py)

        return integrate.quad(integrand, -15.0, 15.0, limit=200)[0]

    return 0.5 * part(1.0) + 0.5 * part(-1.0)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
        caps.append(_bpsk_mutual_info(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=400):
    """使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    for _ in range(60):
        mid = (lo + hi) / 2.0
        c = compute_bpsk_capacity(np.array([mid]), rate)[0]
        if c > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None,
    xlabel="Eb/N0 (dB)", ylabel="BLER",
):
    """绘制 BLER 曲线（对数纵轴）"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
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
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_use = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, "
                f"R={K_use / N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_use}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - K_use}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
