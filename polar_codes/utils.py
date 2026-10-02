"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

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


def _log2_1_plus_exp(m):
    """数值稳定的 log2(1 + exp(-m))"""
    m = np.asarray(m, dtype=np.float64)
    out = np.empty_like(m)
    pos = m > 0
    out[pos] = m[pos] / np.log(2) + np.log1p(np.exp(-m[pos])) / np.log(2)
    out[~pos] = np.log1p(np.exp(m[~pos])) / np.log(2)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))

        def integrand(y):
            return _log2_1_plus_exp(2.0 * snr * y) * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def _mc_bpsk_capacity(eb_n0_db, rate, block_size=4000, seed=0):
    """蒙特卡洛估计 BPSK-AWGN 互信息（bits/channel use）"""
    from channel import awgn_channel, bpsk_modulate, eb_n0_to_sigma

    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, block_size)
    y = awgn_channel(bpsk_modulate(bits), sigma, rng)
    llr = 2.0 * y / (sigma ** 2)
    p_match = np.exp(-bits * llr) / (1.0 + np.exp(-llr))
    p_match = np.clip(p_match, 1e-12, 1.0)
    return float(1.0 + np.mean(np.log2(p_match)))


def find_capacity_limit(rate, eb_n0_range=(-1, 6), num_points=80):
    """使 BPSK 互信息等于码率 R 的 Eb/N0（dB），用于图中香农限参考"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([_mc_bpsk_capacity(eb, rate) for eb in eb_grid])
    for i in range(len(eb_grid) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0 and abs(c1 - c0) > 1e-6:
            t = (rate - c0) / (c1 - c0)
            return float(eb_grid[i] + t * (eb_grid[i + 1] - eb_grid[i]))
    return float(eb_grid[np.argmin(np.abs(caps - rate))])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER 曲线（对数纵轴）"""
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity limit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
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
            if K is None:
                K_loc = N // 2
            else:
                K_loc = K
            info_idx, frozen_idx, _ = ga_construction(N, K_loc, design_eb_n0_db)
            rate = K_loc / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_loc}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {K_loc}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {N - K_loc}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
