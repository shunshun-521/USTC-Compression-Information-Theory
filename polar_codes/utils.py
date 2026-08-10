"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
            'avg_decode_time_ms', 'avg_iters',
        ])
        for r in results:
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
    results = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'eb_n0_db': float(row['eb_n0_db']),
                'bler': float(row['bler']),
                'ber': float(row['ber']),
                'num_errors': int(row['num_errors']),
                'num_frames': int(row['num_frames']),
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000.0,
                'avg_iters': float(row['avg_iters']) if row['avg_iters'] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
  """
  计算 BPSK 离散输入 AWGN 信道容量（bits/channel use）。

  C = 1 - (2/sqrt(pi)) * integral_0^inf exp(-x^2) log2(1 + exp(-SNR*x^2)) dx
  其中 SNR = Es/N0 = 2*R*10^(Eb/N0/10)。
  """
  capacities = []
  for eb_n0_db in eb_n0_db_list:
    snr_linear = 2 * rate * (10 ** (eb_n0_db / 10.0))

    def integrand(x):
      val = -snr_linear * x * x
      if val < -700:
        inner = 0.0
      elif val > 700:
        inner = val / np.log(2)
      else:
        inner = np.log2(1 + np.exp(val))
      return inner * np.exp(-x * x)

    integral, _ = integrate.quad(integrand, 0, 50)
    c = 1 - 2 * integral / np.sqrt(np.pi)
    capacities.append(c)
  return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 15), num_points=200):
  """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
  lo, hi = eb_n0_range
  for _ in range(60):
    mid = (lo + hi) / 2
    cap = compute_bpsk_capacity([mid], rate)[0]
    if cap >= rate:
      hi = mid
    else:
      lo = mid
  return (lo + hi) / 2


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1,
                   label=f'Capacity limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            elif isinstance(K, (list, tuple)):
                K_n = K[N_list.index(N)]
            else:
                K_n = K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            rate = K_n / N
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
