"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
  """将仿真结果保存为 CSV 文件"""
  os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
  with open(filepath, "w", newline="", encoding="utf-8") as f:
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
          "" if r["avg_iters"] is None else f"{r['avg_iters']:.3f}",
        ]
      )


def load_results_csv(filepath):
  """从 CSV 文件加载仿真结果"""
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
          if not row.get("avg_iters") or row["avg_iters"] == ""
          else float(row["avg_iters"]),
        }
      )
  return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
  """计算 BPSK 离散输入信道容量（bits/channel use）"""
  caps = []
  for eb_n0_db in eb_n0_db_list:
    snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))

    def integrand(y):
      return np.log2(1.0 + np.exp(-2.0 * snr * y ** 2)) * np.exp(-y ** 2) / np.sqrt(np.pi)

    val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    caps.append(1.0 - val)
  return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
  """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
  lo, hi = eb_n0_range
  grid = np.linspace(lo, hi, num_points)
  caps = compute_bpsk_capacity(grid, rate)
  idx = np.argmin(np.abs(caps - rate))
  if idx == 0 or idx == len(grid) - 1:
    for _ in range(80):
      mid = (lo + hi) / 2.0
      cap = compute_bpsk_capacity([mid], rate)[0]
      if cap > rate:
        hi = mid
      else:
        lo = mid
    return (lo + hi) / 2.0
  return float(grid[idx])


def plot_bler_curves(
  results_dict,
  title,
  save_path,
  shannon_limit_db=None,
  xlabel="Eb/N0 (dB)",
  ylabel="BLER",
):
  """绘制 BLER-Eb/N0 曲线"""
  os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 5))
  for label, results in results_dict.items():
    eb = [r["eb_n0_db"] for r in results]
    bler = [max(r["bler"], 1e-8) for r in results]
    ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)
  if shannon_limit_db is not None:
    ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(True, which="both", alpha=0.3)
  ax.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(save_path, dpi=150)
  pdf_path = os.path.splitext(save_path)[0] + ".pdf"
  plt.savefig(pdf_path)
  plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
  """保存信息位/冻结位集合"""
  os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
  with open(save_path, "w", encoding="utf-8") as f:
    for N in N_list:
      k_val = K if K is not None else N // 2
      rate = k_val / N
      info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate)
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
