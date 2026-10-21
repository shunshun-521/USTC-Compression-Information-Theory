"""工具函数：结果保存、绘图、容量计算、CRC"""
import csv
import os
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)，返回信息位 + 校验位"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC_POLY[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def save_results_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time",
        "avg_iters",
    ]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = dict(r)
            if row.get("avg_iters") is None:
                row["avg_iters"] = ""
            w.writerow(row)


def load_results_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        out = []
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
    """BPSK-AWGN 互信息（bits / channel use），与 channel.eb_n0_to_sigma 一致。"""
    caps = []
    for eb in eb_n0_db_list:
        sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-eb / 20.0))
        grid = np.linspace(-1.0 - 8.0 * sigma, 1.0 + 8.0 * sigma, 40001)
        dy = grid[1] - grid[0]
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        p0 = norm * np.exp(-0.5 * ((grid - 1.0) / sigma) ** 2)
        p1 = norm * np.exp(-0.5 * ((grid + 1.0) / sigma) ** 2)
        denom = p0 + p1 + 1e-300
        integrand = 0.5 * p0 * np.log2(2.0 * p0 / denom) + 0.5 * p1 * np.log2(
            2.0 * p1 / denom
        )
        mi = float(np.sum(integrand) * dy)
        caps.append(max(0.0, mi))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    from scipy.optimize import brentq

    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)

    def cap_at(eb):
        return float(compute_bpsk_capacity([eb], rate)[0] - rate)

    for i in range(len(grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            return float(brentq(cap_at, grid[i], grid[i + 1]))
    return float(grid[np.argmin(np.abs(caps - rate))])


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)
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
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            K_use = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_use, design_eb_n0_db)
            rate = K_use / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_use}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
