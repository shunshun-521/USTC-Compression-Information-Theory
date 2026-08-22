"""工具函数：结果保存、绘图、容量计算"""
import numpy as np
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import integrate
from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
                         'avg_decode_time_ms', 'avg_iters'])
        for r in results:
            avg_iters = r['avg_iters'] if r['avg_iters'] is not None else ''
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                avg_iters if avg_iters != '' else '',
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
    results = []
    with open(filepath, 'r') as f:
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
    计算 BPSK 离散输入信道容量（bits/channel use）。
 
    C = 1 - E_{y}[log2(1 + e^{-2*s*y})]，其中 s = SNR = 2R * 10^{Eb/N0/10}
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb_n0_db / 10.0))
        sigma = 1.0 / np.sqrt(snr)

        def integrand(y):
            return np.log2(1 + np.exp(-2 * snr * y)) * np.exp(-y ** 2 / (2 * sigma ** 2))

        integral, _ = integrate.quad(integrand, -10 * sigma, 10 * sigma)
        c = 1.0 - integral / np.sqrt(2 * np.pi * sigma ** 2)
        capacities.append(c)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    """
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_n0_vals, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx > 0 and idx < len(eb_n0_vals) - 1:
        # Linear interpolation
        c0, c1 = caps[idx - 1], caps[idx + 1]
        e0, e1 = eb_n0_vals[idx - 1], eb_n0_vals[idx + 1]
        if c0 != c1:
            return e0 + (rate - c0) * (e1 - e0) / (c1 - c0)
    return eb_n0_vals[idx]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, (label, results) in enumerate(results_dict.items()):
        eb_n0 = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb_n0, bler, marker=markers[i % len(markers)],
                    label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(x=shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {K_n}):\n')
            f.write(np.array2string(info_idx, separator=' ', max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {N - K_n}):\n')
            f.write(np.array2string(frozen_idx, separator=' ', max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
