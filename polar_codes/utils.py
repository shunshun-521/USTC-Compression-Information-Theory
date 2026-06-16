"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, r):
    reg = 0
    for b in bits:
        reg ^= int(b) << (r - 1)
        for _ in range(8):
            if reg & (1 << (r - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << r) - 1)
            else:
                reg = (reg << 1) & ((1 << r) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def crc_check_bits(info_bits, crc_length=8):
    """仅对信息比特计算 CRC 并校验（不含冻结位）"""
    encoded = crc_encode(info_bits, crc_length)
    return crc_check(encoded, crc_length)


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
    """从 CSV 加载仿真结果"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
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
                    "avg_iters": None
                    if row.get("avg_iters", "") == ""
                    else float(row["avg_iters"]),
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）"""
    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))

        def integrand(y):
            py = np.exp(-(y ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
            val = -snr * (y ** 2)
            log_term = np.log2(1.0 + np.exp(val))
            return log_term * py

        cap_loss, _ = integrate.quad(integrand, -15.0, 15.0, limit=200)
        caps.append(1.0 - cap_loss)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        return float(eb_grid[idx])
    if caps[idx] >= rate:
        i0, i1 = idx - 1, idx
    else:
        i0, i1 = idx, idx + 1
    c0, c1 = caps[i0], caps[i1]
    e0, e1 = eb_grid[i0], eb_grid[i1]
    if c1 == c0:
        return float(e0)
    return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=info_idx.size, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=frozen_idx.size, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
