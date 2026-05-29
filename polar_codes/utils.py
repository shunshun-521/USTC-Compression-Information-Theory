"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from construction import ga_construction


def save_results_csv(results, filepath):
    """保存仿真结果为 CSV"""
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
                    f"{r['bler']:.6e}",
                    f"{r['ber']:.6e}",
                    r["num_errors"],
                    r["num_frames"],
                    f"{r['avg_decode_time'] * 1000:.6f}",
                    "" if r["avg_iters"] is None else f"{r['avg_iters']:.2f}",
                ]
            )


def load_results_csv(filepath):
    """从 CSV 加载结果"""
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
                    "avg_iters": float(row["avg_iters"])
                    if row.get("avg_iters", "")
                    else None,
                }
            )
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """BPSK 离散输入信道容量（bits/channel use）"""
    from scipy import integrate

    caps = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            return np.log2(1.0 + np.exp(-2.0 * snr * y**2)) * np.exp(-y**2) / np.sqrt(
                np.pi
            )

        val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
        caps.append(1.0 - val)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 容量等于码率 R 的 Eb/N0（dB）"""
    try:
        from scipy import integrate
    except ImportError:
        return 0.79 * (10 ** ((-10 * np.log10(rate)) / 10))  # 粗略近似

    lo, hi = eb_n0_range

    def capacity_at(eb_db):
        snr = 2.0 * rate * (10.0 ** (eb_db / 10.0))

        def integrand(y):
            return np.log2(1.0 + np.exp(-2.0 * snr * y**2)) * np.exp(-y**2) / np.sqrt(
                np.pi
            )

        val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
        return 1.0 - val - rate

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if capacity_at(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None, xlabel="Eb/N0 (dB)", ylabel="BLER"
):
    """绘制 BLER-Eb/N0 曲线（PNG + PDF）"""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-7) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(
            shannon_limit_db,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Capacity limit ({shannon_limit_db:.2f} dB)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate=rate)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N, separator=" ") + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, separator=" ") + "\n")
            f.write("-" * 53 + "\n")
