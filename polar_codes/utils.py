"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from scipy import integrate

from construction import ga_construction


def _crc_remainder(bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC 编码（系统位 + 校验位）。r=8: 0x07, r=16: 0x8005"""
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = _crc_remainder(info_bits + [0] * crc_length, crc_length, poly)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(info_bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC（与 crc_encode 成对验证）。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV。"""
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
                    "avg_iters": r["avg_iters"] if r["avg_iters"] is not None else "",
                }
            )


def load_results_csv(filepath):
    """从 CSV 加载结果。"""
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


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入 AWGN 信道容量（bits/实信道使用）。"""
    caps = []
    for eb_n0_db in eb_n0_db_list:
        gamma = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            t = -2.0 * np.sqrt(gamma) * y
            if t > 50:
                log_term = 0.0
            elif t < -50:
                log_term = -t / np.log(2.0)
            else:
                log_term = np.log1p(np.exp(t)) / np.log(2.0)
            return log_term * np.exp(-(y ** 2) / 2.0) / np.sqrt(2.0 * np.pi)

        val, _ = integrate.quad(integrand, -30.0, 30.0)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=200):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）近似值。"""
    if rate <= 0 or rate >= 1:
        return 10.0
    lin = (2 ** (2 * rate) - 1) / (2 * rate)
    if lin <= 0:
        return eb_n0_range[1]
    return float(10.0 * np.log10(lin))


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER-Eb/N0 曲线。"""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        x = [r["eb_n0_db"] for r in results]
        y = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(x, y, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

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
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
