"""
工具函数：CRC、绘图、结果保存
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "eb_n0_db": row["eb_n0_db"],
                    "bler": row["bler"],
                    "ber": row["ber"],
                    "num_errors": row["num_errors"],
                    "num_frames": row["num_frames"],
                    "avg_decode_time_ms": row["avg_decode_time"] * 1000.0,
                    "avg_iters": row["avg_iters"] if row["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
                    "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
                }
            )
    return results


def _bpsk_capacity_scalar(eb_n0_db, rate):
    """单点 BPSK 容量（bits/channel use）。"""
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

    def integrand(y):
        return np.log2(1.0 + np.exp(-2.0 * snr * y)) * np.exp(-0.5 * y * y)

    val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    val /= np.sqrt(2.0 * np.pi)
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量。"""
    eb_n0_db_list = np.asarray(eb_n0_db_list, dtype=np.float64)
    return np.array([_bpsk_capacity_scalar(e, rate) for e in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_vals, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_vals) - 1:
        lo, hi = eb_n0_range[0], eb_n0_range[1]
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if _bpsk_capacity_scalar(mid, rate) < rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    return float(eb_vals[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

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
    """保存各码长的信息位/冻结位集合。"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = N // 2 if K is None else K
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ") + "\n")
            f.write("-" * 53 + "\n")


# ==================== CRC 工具 ====================

def _bits_to_bytes(bits):
    """MSB-first 比特打包为字节。"""
    n = len(bits)
    pad = (-n) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        out.append(byte)
    return bytes(out)


def _crc8_bytes(data_bytes):
    """CRC-8 (poly 0x07)。"""
    crc = 0
    for value in data_bytes:
        crc ^= value
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc8_bits(bits):
    return _crc8_bytes(_bits_to_bytes(bits))


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length != 8:
        raise ValueError("当前实现支持 crc_length=8")
    info_bits = np.asarray(info_bits, dtype=np.int8)
    crc_val = _crc8_bits(info_bits)
    crc_bits = np.array([(crc_val >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length != 8:
        raise ValueError("当前实现支持 crc_length=8")
    bits = np.asarray(bits, dtype=np.int8)
    return _crc8_bits(bits) == 0
