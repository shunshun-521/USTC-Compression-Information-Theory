"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
            'avg_decode_time_ms', 'avg_iters'
        ])
        for r in results:
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.4f}",
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
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
    C = 1 - E[log2(1 + exp(-2*snr*Y^2))], Y ~ N(0,1)
    """
    capacities = []
    y = np.linspace(-6, 6, 10000)
    weights = np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
        val = np.clip(-2.0 * snr * y ** 2, -50, 50)
        integrand = weights * np.log2(1.0 + np.exp(val))
        capacities.append(max(1.0 - np.trapezoid(integrand, y), 0.0))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_n0_vals, rate)
    idx = np.argmin(np.abs(caps - rate))
    if caps[idx] >= rate and idx > 0:
        return eb_n0_vals[idx]
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    for _ in range(50):
        mid = (lo + hi) / 2
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for idx, (label, results) in enumerate(results_dict.items()):
        eb_n0 = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb_n0, bler, marker=markers[idx % len(markers)],
                    label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(x=shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
