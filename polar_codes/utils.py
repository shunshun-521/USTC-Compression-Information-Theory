"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np

try:
  import matplotlib.pyplot as plt
  HAS_MPL = True
except ImportError:
  HAS_MPL = False

from construction import ga_construction


def save_results_csv(results, filepath):
  """将仿真结果保存为 CSV 文件"""
  os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
  with open(filepath, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
      "eb_n0_db", "bler", "ber", "num_errors", "num_frames",
      "avg_decode_time_ms", "avg_iters",
    ])
    for r in results:
      writer.writerow([
        f"{r['eb_n0_db']:.4f}",
        f"{r['bler']:.6e}",
        f"{r['ber']:.6e}",
        r["num_errors"],
        r["num_frames"],
        f"{r['avg_decode_time'] * 1000:.6f}",
        "" if r["avg_iters"] is None else f"{r['avg_iters']:.4f}",
      ])


def load_results_csv(filepath):
  """从 CSV 文件加载仿真结果"""
  results = []
  with open(filepath, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      results.append({
        "eb_n0_db": float(row["eb_n0_db"]),
        "bler": float(row["bler"]),
        "ber": float(row["ber"]),
        "num_errors": int(row["num_errors"]),
        "num_frames": int(row["num_frames"]),
        "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
        "avg_iters": None if row["avg_iters"] == "" else float(row["avg_iters"]),
      })
  return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
  """计算 BPSK 离散输入信道容量（bits/channel use）"""
  eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
  capacities = []
  rng = np.random.default_rng(0)
  n_mc = 200000
  x = rng.choice(np.array([-1.0, 1.0]), n_mc)
  noise = rng.normal(0.0, 1.0, n_mc)

  for eb in eb_n0_db_list:
    snr = 2.0 * rate * (10 ** (eb / 10.0))
    sigma = 1.0 / np.sqrt(snr)
    y = x + noise * sigma
    p0 = np.exp(-0.5 * ((y - 1.0) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    p1 = np.exp(-0.5 * ((y + 1.0) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    py = 0.5 * (p0 + p1)
    px_y = np.where(x > 0, p0, p1)
    capacities.append(float(np.mean(np.log2(np.maximum(px_y / py, 1e-300)))))
  return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-1, 6), num_points=200):
  """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
  lo, hi = eb_n0_range
  for _ in range(60):
    mid = 0.5 * (lo + hi)
    cap = compute_bpsk_capacity(mid, rate)[0]
    if cap < rate:
      lo = mid
    else:
      hi = mid
  return 0.5 * (lo + hi)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel="Eb/N0 (dB)", ylabel="BLER"):
  """绘制 BLER-Eb/N0 曲线"""
  os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

  if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
      eb = [r["eb_n0_db"] for r in results]
      bler = [max(r["bler"], 1e-8) for r in results]
      ax.semilogy(eb, bler, "o-", label=label, linewidth=1.5, markersize=4)
    if shannon_limit_db is not None:
      ax.axvline(shannon_limit_db, color="gray", linestyle="--", linewidth=1.2,
                 label=f"Capacity limit ({shannon_limit_db:.2f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
  else:
    _save_simple_svg(results_dict, title, save_path, shannon_limit_db)


def _save_simple_svg(results_dict, title, save_path, shannon_limit_db):
  """matplotlib 不可用时的简易 SVG 输出"""
  lines = [f"<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500'>",
           f"<text x='20' y='25'>{title}</text>"]
  y = 40
  for label, results in results_dict.items():
    pts = " ".join(f"{20+r['eb_n0_db']*60},{400-np.log10(max(r['bler'],1e-8))*40}" for r in results)
    lines.append(f"<polyline points='{pts}' fill='none' stroke='black'/>")
    lines.append(f"<text x='20' y='{y}'>{label}</text>")
    y += 16
  if shannon_limit_db is not None:
    x = 20 + shannon_limit_db * 60
    lines.append(f"<line x1='{x}' y1='30' x2='{x}' y2='450' stroke='gray' stroke-dasharray='4'/>")
  lines.append("</svg>")
  with open(save_path.replace(".png", ".svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
  """保存各码长的信息位/冻结位集合"""
  os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
  with open(save_path, "w", encoding="utf-8") as f:
    for N in N_list:
      k_val = K if K is not None else int(N * rate)
      info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db, rate=k_val / N)
      f.write("=" * 53 + "\n")
      f.write(f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={k_val/N:.4f}\n")
      f.write("=" * 53 + "\n")
      f.write(f"Info indices (all {len(info_idx)}):\n")
      f.write(np.array2string(info_idx, max_line_width=120) + "\n")
      f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
      f.write(np.array2string(frozen_idx, max_line_width=120) + "\n")
      f.write("-" * 53 + "\n")
