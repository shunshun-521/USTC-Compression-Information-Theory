"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


_CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _bits_to_bytes(bits):
    """将比特序列按 MSB-first 打包为字节"""
    bits = [int(b) for b in bits]
    nbytes = (len(bits) + 7) // 8
    data = bytearray(nbytes)
    for i, bit in enumerate(bits):
        data[i // 8] |= bit << (7 - i % 8)
    return bytes(data)


def _crc8_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x8005) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _crc_remainder(bits, crc_length):
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_bytes(data)
    if crc_length == 16:
        return _crc16_bytes(data)
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 编码，将校验位附加到信息比特后"""
    if crc_length not in _CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length not in _CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time, avg_iters
    """
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
            avg_iters = r.get("avg_iters")
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.4f}",
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000.0:.6f}",
                    "" if avg_iters is None else f"{avg_iters:.4f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                        float(row["avg_iters"]) if row.get("avg_iters", "") else None
                    ),
                }
            )
    return results


def _bpsk_capacity_at_eb_n0(eb_n0_db, rate):
    """单点 BPSK 容量（bits/channel use）"""
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

    def integrand(y):
        z = -2.0 * snr * y * y
        if z > 40.0:
            log_term = z / np.log(2.0)
        elif z < -40.0:
            log_term = 0.0
        else:
            log_term = np.log2(1.0 + np.exp(z))
        return log_term * np.exp(-0.5 * y * y)

    val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    val /= np.sqrt(2.0 * np.pi)
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。

    C = 1 - E_{y}[log2(1 + e^{-2*s*y})]，其中 s = SNR = 2R * 10^{Eb/N0/10}
    """
    return np.array([_bpsk_capacity_at_eb_n0(eb, rate) for eb in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-3, 5), num_points=200):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    这是香农限，用于在 BLER 图中标注参考竖线。
    """
    lo, hi = eb_n0_range
    caps_lo = _bpsk_capacity_at_eb_n0(lo, rate) - rate
    caps_hi = _bpsk_capacity_at_eb_n0(hi, rate) - rate
    if caps_lo * caps_hi > 0:
        eb_grid = np.linspace(lo, hi, num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        idx = int(np.argmin(np.abs(caps - rate)))
        return float(eb_grid[idx])

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _bpsk_capacity_at_eb_n0(mid, rate) > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-8) for r in results]
        ax.semilogy(
            eb,
            bler,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            label=label,
        )

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Shannon limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(
                np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n"
            )
            f.write("-" * 53 + "\n")
