"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def crc_encode_bits(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int64).ravel()
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = 0x8005
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) & 0xFFFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check_bits(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int64).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode_bits(payload, crc_length)
    return np.array_equal(expected, bits)


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            w.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.6f}",
                "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
            ])


def load_results_csv(filepath):
    results = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": None if row["avg_iters"] == "" else float(row["avg_iters"]),
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))

        def integrand(y):
            t = -2.0 * snr * y
            t = np.clip(t, -700.0, 700.0)
            return np.log2(1.0 + np.exp(t)) * np.exp(-0.5 * y * y)

        val, _ = integrate.quad(integrand, -np.inf, np.inf)
        val /= np.sqrt(2.0 * np.pi)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    lo, hi = eb_n0_range
    grid = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(grid) - 1:
        return float(grid[idx])
    g0, g1 = grid[idx - 1], grid[idx + 1]
    c0, c1 = caps[idx - 1], caps[idx + 1]
    if c1 == c0:
        return float(grid[idx])
    t = (rate - c0) / (c1 - c0)
    return float(g0 + t * (g1 - g0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    plt.figure(figsize=(8, 5))
    for label, results in results_dict.items():
        xs = [r["eb_n0_db"] for r in results]
        ys = [max(r["bler"], 1e-6) for r in results]
        plt.semilogy(xs, ys, "o-", label=label)
    if shannon_limit_db is not None:
        plt.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_override, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K = N // 2 if K_override is None else K_override
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db, rate)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
