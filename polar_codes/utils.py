"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from construction import ga_construction


def _crc_register_bits(bits, crc_length):
    if crc_length == 8:
        poly, top = 0x07, 0x80
        reg = 0
        for bit in bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & top:
                    reg = ((reg << 1) & 0xFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFF
        return reg
    if crc_length == 16:
        poly, top = 0x8005, 0x8000
        reg = 0
        for bit in bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & top:
                    reg = ((reg << 1) & 0xFFFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_register_bits(info_bits, crc_length)
    if crc_length == 8:
        crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    else:
        crc_bits = np.array([(reg >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


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
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            t = np.clip(-snr * y * y, -700.0, 700.0)
            return np.log2(1.0 + np.exp(t)) * np.exp(-0.5 * y * y) / np.sqrt(2.0 * np.pi)

        val, _ = integrate.quad(integrand, -np.inf, np.inf)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """
    BPSK-AWGN 下码率 R 对应的参考 Eb/N0（dB），用于 BLER 图竖线。
    若数值互信息曲线始终高于 R，则退回经典近似 (2^(2R)-1)/(2R)。
    """
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    if np.all(caps > rate):
        x = (2.0 ** (2.0 * rate) - 1.0) / (2.0 * rate)
        return float(10.0 * np.log10(max(x, 1e-12)))
    diff = caps - rate
    idx = np.where(diff >= 0)[0]
    if len(idx) == 0 or idx[0] == 0:
        x = (2.0 ** (2.0 * rate) - 1.0) / (2.0 * rate)
        return float(10.0 * np.log10(max(x, 1e-12)))

    def obj(eb_db):
        return compute_bpsk_capacity(np.array([eb_db]), rate)[0] - rate

    lo, hi = grid[idx[0] - 1], grid[idx[0]]
    return float(brentq(obj, lo, hi))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate=K_n / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={K_n/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
