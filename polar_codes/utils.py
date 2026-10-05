"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from construction import ga_construction


def save_results_csv(results, filepath):
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
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
                ]
            )


def load_results_csv(filepath):
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
    """BPSK-AWGN 信道容量（bits/channel use）"""
    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            # BPSK 输入 AWGN 互信息（标准高斯积分形式）
            t = snr * y * y
            if t > 40:
                inner = -t * np.log2(np.e)
            elif t < 1e-12:
                inner = 0.0
            else:
                inner = np.log2(1.0 + np.exp(-t))
            return inner * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=100)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(0.0, 5.0), num_points=400):
    """找到 BPSK 容量 C=rate 时的 Eb/N0（dB）"""
    from scipy.optimize import brentq

    def cap_minus_r(eb_db):
        return compute_bpsk_capacity([eb_db], rate)[0] - rate

    lo, hi = eb_n0_range
    caps_lo = cap_minus_r(lo)
    caps_hi = cap_minus_r(hi)
    if caps_lo * caps_hi > 0:
        # 回退：网格搜索
        eb_grid = np.linspace(lo, hi, num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        idx = np.argmin(np.abs(caps - rate))
        return float(eb_grid[idx])
    return float(brentq(cap_minus_r, lo, hi))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        plt.semilogy(xs, ys, marker="o", linewidth=1.5, label=label)

    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon R (≈{shannon_limit_db:.2f} dB)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                k = N // 2
            else:
                k = K
            rate = k / N
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(
                f"Info indices (all {len(info_idx)}):\n"
                f"{np.array2string(info_idx, max_line_width=120)}\n"
            )
            f.write(
                f"Frozen indices (all {len(frozen_idx)}):\n"
                f"{np.array2string(frozen_idx, max_line_width=120)}\n"
            )
            f.write("-" * 53 + "\n")
