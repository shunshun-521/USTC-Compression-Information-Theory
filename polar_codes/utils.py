"""工具函数：结果保存、绘图、容量计算、CRC"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


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
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                    if row["avg_iters"] == ""
                    else float(row["avg_iters"]),
                }
            )
    return results


def bpsk_capacity(es_n0_linear):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    C = 1 - E_x[log2(1 + exp(-2*EsN0*x^2))], x ~ N(0,1)
    """

    def integrand(x):
        return (
            np.log2(1.0 + np.exp(-2.0 * es_n0_linear * x ** 2))
            * np.exp(-0.5 * x ** 2)
            / np.sqrt(2.0 * np.pi)
        )

    val, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """返回各 Eb/N0 点对应的 BPSK 信道容量"""
    caps = []
    for eb in eb_n0_db_list:
        es_n0 = (10 ** (eb / 10.0)) * rate
        caps.append(bpsk_capacity(es_n0))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range
    eb_vals = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(eb_vals, rate)
    idx = np.argmin(np.abs(caps - rate))
    if caps[idx] > rate and idx > 0:
        lo2, hi2 = eb_vals[idx - 1], eb_vals[idx]
    elif caps[idx] < rate and idx < len(eb_vals) - 1:
        lo2, hi2 = eb_vals[idx], eb_vals[idx + 1]
    else:
        return float(eb_vals[idx])
    for _ in range(50):
        mid = (lo2 + hi2) / 2.0
        cap = bpsk_capacity((10 ** (mid / 10.0)) * rate)
        if cap > rate:
            hi2 = mid
        else:
            lo2 = mid
    return (lo2 + hi2) / 2.0


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
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
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
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
