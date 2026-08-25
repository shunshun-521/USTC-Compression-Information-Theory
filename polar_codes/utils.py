"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from construction import ga_construction


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


def compute_bpsk_capacity(eb_n0_db_list, rate):
  """
  计算 BPSK 离散输入信道容量（bits/channel use）。
  C = 1 - (1/pi) * int_0^{pi/2} log2(1 + SNR*sin^2(phi)) d(phi)
  SNR = Es/N0 = 2*R*10^{Eb/N0/10}
  """
  capacities = []
  for eb_n0_db in eb_n0_db_list:
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    phi = np.linspace(0.0, np.pi / 2.0, 2000)
    integrand = np.log2(1.0 + snr * np.sin(phi) ** 2)
    val = np.trapezoid(integrand, phi) / np.pi
    capacities.append(max(0.0, 1.0 - val))
  return np.asarray(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=200):
  """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
  eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
  caps = compute_bpsk_capacity(eb_grid, rate)

  for i in range(1, len(eb_grid)):
    if caps[i - 1] <= rate <= caps[i] or caps[i - 1] >= rate >= caps[i]:
      e0, e1 = eb_grid[i - 1], eb_grid[i]
      c0, c1 = caps[i - 1], caps[i]
      if c1 == c0:
        return float(e0)
      return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))

  return float(eb_grid[np.argmin(np.abs(caps - rate))])


def plot_bler_curves(
    results_dict,
    title,
    save_path,
    shannon_limit_db=None,
    xlabel="Eb/N0 (dB)",
    ylabel="BLER",
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r["eb_n0_db"] for r in results]
        bler = [max(r["bler"], 1e-6) for r in results]
        ax.semilogy(eb, bler, "o-", label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2, label="Shannon limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
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
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            rate = k_val / N
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
