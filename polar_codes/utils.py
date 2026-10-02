"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    rem = reg
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


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
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.4e}",
                    f"{r['ber']:.4e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r.get("avg_iters") is None else f"{r['avg_iters']:.4f}",
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（Monte Carlo 估计）。"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        # 对称 BPSK BI-AWGN：C ≈ 1 - E[log2(1+exp(-2*snr*Y))], Y~N(0,1)
        y = np.linspace(-8.0, 8.0, 8001)
        w = np.exp(-0.5 * y * y)
        w /= np.sum(w)
        z = -2.0 * snr * y
        term = np.log2(1.0 + np.exp(np.clip(z, -80, 80)))
        caps.append(float(np.clip(1.0 - np.sum(term * w), 0.0, 1.0)))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """使 log2(1+2R*Eb/N0) = R 的 Eb/N0（BPSK 常用 Shannon 参考线）。"""
    lo, hi = eb_n0_range

    def awgn_spectral(rate_val, eb_db):
        snr = 2.0 * rate_val * (10.0 ** (eb_db / 10.0))
        return np.log2(1.0 + snr)

    if awgn_spectral(rate, lo) > rate:
        return float(lo)
    if awgn_spectral(rate, hi) < rate:
        return float(hi)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if awgn_spectral(rate, mid) >= rate:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


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
    plt.grid(True, which="both", alpha=0.4)
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
            K_use = int(N * rate) if K is None else K
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db, rate=K_use / N)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={K_use/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
