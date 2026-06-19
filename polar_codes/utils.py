"""工具函数：结果保存、绘图、容量计算、CRC"""
import csv
import os

import numpy as np

from construction import ga_construction

CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
            "avg_decode_time_ms", "avg_iters",
        ])
        for r in results:
            avg_iters = r.get("avg_iters")
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r["num_errors"],
                r["num_frames"],
                f"{r['avg_decode_time'] * 1000:.4f}",
                "" if avg_iters is None else f"{avg_iters:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                "avg_iters": float(row["avg_iters"]) if row["avg_iters"] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK-AWGN 信道容量（bits/channel use）。"""
    capacities = []
    y = np.linspace(-15, 15, 20001)
    norm = 2 * np.sqrt(2 * np.pi)
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * (10 ** (eb_n0_db / 10.0))
        p0 = np.exp(-0.5 * (y - np.sqrt(snr)) ** 2)
        p1 = np.exp(-0.5 * (y + np.sqrt(snr)) ** 2)
        s = p0 + p1
        term0 = np.where(s > 0, p0 * np.log2(2 * p0 / s), 0.0)
        term1 = np.where(s > 0, p1 * np.log2(2 * p1 / s), 0.0)
        capacities.append(np.trapezoid(term0 + term1, y) / norm)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    for i in range(len(eb_grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            c0, c1 = caps[i], caps[i + 1]
            e0, e1 = eb_grid[i], eb_grid[i + 1]
            if c1 == c0:
                return float(e0)
            return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))
    idx = int(np.argmin(np.abs(caps - rate)))
    return float(eb_grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    """绘制 BLER-Eb/N0 曲线。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--",
                   label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

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
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=len(info_idx)) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=len(frozen_idx)) + "\n")
            f.write("-" * 53 + "\n")
