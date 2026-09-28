"""工具函数：CRC、绘图、结果保存、log 域运算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def logdomain_sum(x, y):
    """log(exp(x)+exp(y)) 的稳定实现"""
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def logdomain_diff(x, y):
    """log(exp(x)-exp(y))，x>=y"""
    if x >= y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=int)
    check = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(check[-crc_length:], bits[-crc_length:])


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
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
            w.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"] * 1000.0,
                    "" if r["avg_iters"] is None else r["avg_iters"],
                ]
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="") as f:
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
                    "avg_iters": None if row["avg_iters"] == "" else float(row["avg_iters"]),
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            x = -2.0 * snr * y
            x = np.clip(x, -700.0, 700.0)
            return np.log2(1.0 + np.exp(x)) * np.exp(-(y ** 2) / 2.0)

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        caps.append(1.0 - val / np.sqrt(2.0 * np.pi))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5.0, 20.0), num_points=1000):
    """求使 C=rate 的 Eb/N0 (dB)"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.where(caps >= rate)[0]
    if len(idx) == 0:
        return float(grid[-1])
    return float(grid[idx[0]])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        plt.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon (R) @ {shannon_limit_db:.2f} dB")
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


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            info, frozen, _ = ga_construction(N, K_n, design_eb_n0_db, rate=K_n / N)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={K_n/N:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n{np.array2string(info, max_line_width=120)}\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n{np.array2string(frozen, max_line_width=120)}\n")
            f.write("-" * 53 + "\n")
