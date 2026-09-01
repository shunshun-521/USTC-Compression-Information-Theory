"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None

import matplotlib.pyplot as plt

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV"""
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
                    "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载仿真结果"""
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
                    "avg_iters": (
                        float(row["avg_iters"]) if row.get("avg_iters") else None
                    ),
                }
            )
    return results


def _bpsk_capacity_per_snr(snr_linear):
    """
    BPSK 离散输入 AWGN 信道容量（bits/channel use）。
    snr_linear = Es/N0 = 2*R*10^(Eb/N0/10)
    """
    noise_var = 1.0 / max(snr_linear, 1e-12)

    def py(y):
        return (
            0.5 * np.exp(-(y - 1) ** 2 / (2 * noise_var))
            / np.sqrt(2 * np.pi * noise_var)
            + 0.5 * np.exp(-(y + 1) ** 2 / (2 * noise_var))
            / np.sqrt(2 * np.pi * noise_var)
        )

    def entropy_integrand(y):
        p = py(y)
        if p < 1e-300:
            return 0.0
        return -p * np.log2(p)

    if integrate is None:
        return max(0.0, 1.0 - 0.5 * np.log2(1.0 + snr_linear))

    hy, _ = integrate.quad(entropy_integrand, -np.inf, np.inf, limit=200)
    hn = 0.5 * np.log2(2 * np.pi * np.e * noise_var)
    return max(0.0, hy - hn)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """返回各 Eb/N0 对应的 BPSK 容量"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        caps.append(_bpsk_capacity_per_snr(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    a, b = lo, hi
    for _ in range(60):
        mid = (a + b) / 2.0
        snr = 2.0 * rate * (10 ** (mid / 10.0))
        c = _bpsk_capacity_per_snr(snr)
        if c < rate:
            a = mid
        else:
            b = mid
    return (a + b) / 2.0


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)

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
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
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
