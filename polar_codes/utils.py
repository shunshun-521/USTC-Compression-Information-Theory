"""工具函数：CRC、结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# ---------------- CRC ----------------

def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8 if crc_len == 8 else 1):
            if crc_len == 8:
                break
        msb = (reg >> (crc_len - 1)) & 1
        reg = ((reg << 1) | int(b)) & ((1 << crc_len) - 1)
        if msb:
            reg ^= poly & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.uint8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    data, crc_part = bits[:-crc_length], bits[-crc_length:]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(expected, crc_part)


# ---------------- CSV / 绘图 ----------------

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
                "avg_decode_time",
                "avg_iters",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["eb_n0_db"],
                    r["bler"],
                    r["ber"],
                    r["num_errors"],
                    r["num_frames"],
                    r["avg_decode_time"],
                    r.get("avg_iters", ""),
                ]
            )


def load_results_csv(filepath):
    out = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["eb_n0_db"] = float(row["eb_n0_db"])
            row["bler"] = float(row["bler"])
            row["ber"] = float(row["ber"])
            row["num_errors"] = int(row["num_errors"])
            row["num_frames"] = int(row["num_frames"])
            row["avg_decode_time"] = float(row["avg_decode_time"])
            ai = row.get("avg_iters", "")
            row["avg_iters"] = float(ai) if ai not in ("", None) else None
            out.append(row)
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 容量 C(Eb/N0, R)（bits/channel use）。"""
    caps = []
    for eb_db in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb_db / 10.0))
        # 数值积分 E[log2(1+exp(-2*sqrt(snr)*z))], z~N(0,1)
        z = np.linspace(-8, 8, 2000)
        pz = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        integrand = np.log2(1 + np.exp(-2 * np.sqrt(snr) * z))
        caps.append(1.0 - np.trapezoid(integrand * pz, z))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=400):
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    return float(eb_grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="BPSK capacity @ R")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_loc = N // 2
            else:
                K_loc = K
            info, frozen, _ = ga_construction(N, K_loc, design_eb_n0_db)
            rate = K_loc / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_loc}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info)}):\n{info}\n")
            f.write(f"Frozen indices (all {len(frozen)}):\n{frozen}\n")
            f.write("-" * 53 + "\n")
