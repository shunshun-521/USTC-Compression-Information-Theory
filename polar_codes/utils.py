"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

from construction import ga_construction

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
            writer.writerow(
                [
                    f"{r['eb_n0_db']:.2f}",
                    f"{r['bler']:.4e}",
                    f"{r['ber']:.4e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    f"{r['avg_iters']:.1f}" if r["avg_iters"] is not None else "",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
                    "avg_iters": float(row["avg_iters"]) if row.get("avg_iters") else None,
                }
            )
    return results


def _bpsk_mi(snr_linear):
    """BPSK 互信息（每信道使用 bit）。"""
    from scipy import integrate

    scale = np.sqrt(snr_linear)

    def integrand(y):
        p0 = np.exp(-0.5 * (y - scale) ** 2) / np.sqrt(2 * np.pi)
        p1 = np.exp(-0.5 * (y + scale) ** 2) / np.sqrt(2 * np.pi)
        py = 0.5 * (p0 + p1)
        if py < 1e-300:
            return 0.0
        val = 0.0
        if p0 > 0:
            val += 0.5 * p0 * np.log2(p0 / py)
        if p1 > 0:
            val += 0.5 * p1 * np.log2(p1 / py)
        return val

    val, _ = integrate.quad(integrand, -scale * 12, scale * 12, limit=200)
    return val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量。"""
    caps = []
    for eb in eb_n0_db_list:
        es_n0 = rate * (10.0 ** (eb / 10.0))
        caps.append(_bpsk_mi(es_n0))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-1, 10), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    try:
        from scipy import optimize
    except ImportError:
        return 0.0

    def cap_minus_r(eb_db):
        es_n0 = rate * (10.0 ** (eb_db / 10.0))
        return _bpsk_mi(es_n0) - rate

    try:
        root = optimize.brentq(cap_minus_r, eb_n0_range[0], eb_n0_range[1])
        return float(root)
    except ValueError:
        eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        idx = np.argmin(np.abs(caps - rate))
        return float(eb_grid[idx])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            label=f"Capacity limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = int(N * rate)
            else:
                K_n = K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={K_n/N:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
