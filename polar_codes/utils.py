"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode_bits(info_bits, crc_length=8):
    """计算 CRC 校验位，返回长度为 crc_length 的数组"""
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)


def crc_check_bits(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode_bits(info, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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
        for r in results:
            writer.writerow(
                {
                    "eb_n0_db": r["eb_n0_db"],
                    "bler": r["bler"],
                    "ber": r["ber"],
                    "num_errors": r["num_errors"],
                    "num_frames": r["num_frames"],
                    "avg_decode_time_ms": r["avg_decode_time"] * 1000.0,
                    "avg_iters": "" if r["avg_iters"] is None else r["avg_iters"],
                }
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_iters = row["avg_iters"]
            results.append(
                {
                    "eb_n0_db": float(row["eb_n0_db"]),
                    "bler": float(row["bler"]),
                    "ber": float(row["ber"]),
                    "num_errors": int(row["num_errors"]),
                    "num_frames": int(row["num_frames"]),
                    "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                    "avg_iters": None if avg_iters == "" else float(avg_iters),
                }
            )
    return results


def _log2_one_plus_exp(x):
    """数值稳定的 log2(1 + exp(x))"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = x[pos] / np.log(2) + np.log2(1.0 + np.exp(-x[pos]))
    out[~pos] = np.log2(1.0 + np.exp(x[~pos]))
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）"""

    def capacity_at_ebn0(eb_n0_db):
        snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
        s = np.sqrt(snr)
        scale = 1.0 / np.sqrt(2.0 * np.pi)

        def integrand(y):
            return _log2_one_plus_exp(-2.0 * s * y) * np.exp(-0.5 * y * y) * scale

        val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        return 1.0 - val

    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    return np.array([capacity_at_ebn0(x) for x in eb_n0_db_list])


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    eb_vals = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(eb_vals, rate)
    diff = caps - rate
    idx = np.where(diff >= 0)[0]
    if len(idx) == 0:
        return float(hi)
    if idx[0] == 0:
        return float(eb_vals[0])
    i = idx[0]
    x0, x1 = eb_vals[i - 1], eb_vals[i]
    y0, y1 = diff[i - 1], diff[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, (label, results) in enumerate(results_dict.items()):
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(
            eb,
            bler,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            label=label,
        )

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label=f"Shannon limit ({shannon_limit_db:.2f} dB)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = N // 2 if K is None else K
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n")
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, separator=" ", max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, separator=" ", max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
