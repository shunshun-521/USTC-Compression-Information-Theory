"""
工具函数：CRC、结果保存、绘图、容量计算
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from construction import ga_construction


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
        width = 8
    elif crc_length == 16:
        poly = 0x8005
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = 1 << (width - 1)
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & mask:
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(width - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def _crc_remainder(bits, crc_length=8):
    poly = 0x07 if crc_length == 8 else 0x8005
    width = crc_length
    reg = 0
    mask = 1 << (width - 1)
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & mask:
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


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
            avg_iters = "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}"
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{1000.0 * r['avg_decode_time']:.6f}",
                    avg_iters,
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入容量（数值积分，|y| 形式避免溢出）"""
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64)
    y = np.linspace(0.0, 12.0, 12001)
    pdf = np.sqrt(2.0 / np.pi) * np.exp(-0.5 * y * y)
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb / 10.0))
        integrand = np.log2(1.0 + np.exp(-snr * y * y)) * pdf
        val = np.trapezoid(integrand, y)
        caps.append(max(0.0, 1.0 - val))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=500):
    """使 BPSK 容量等于码率 R 的 Eb/N0；若无交点则退回连续 AWGN 香农限"""
    lo, hi = eb_n0_range
    grid = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(grid, rate)
    for i in range(len(grid) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0:
            t = (rate - c0) / (c1 - c0 + 1e-15)
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    snr_lin = (2.0 ** (2.0 * rate) - 1.0) / (2.0 * rate)
    return float(10.0 * np.log10(snr_lin + 1e-15))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
