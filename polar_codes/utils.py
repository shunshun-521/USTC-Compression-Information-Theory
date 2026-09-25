"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from construction import ga_construction
from channel import eb_n0_to_sigma


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
            'avg_decode_time_ms', 'avg_iters',
        ])
        for r in results:
            writer.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.4f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'eb_n0_db': float(row['eb_n0_db']),
                'bler': float(row['bler']),
                'ber': float(row['ber']),
                'num_errors': int(row['num_errors']),
                'num_frames': int(row['num_frames']),
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000.0,
                'avg_iters': None if row['avg_iters'] == '' else float(row['avg_iters']),
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use），与仿真中的 sigma 定义一致。
    C ≈ 1 - E[log2(1 + exp(-|LLR|))], LLR = 2Y/σ²
    """
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    y = np.linspace(-10 * sigma, 10 * sigma, 40000)
    dy = y[1] - y[0]
    p_y = np.exp(-y ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    llr = np.abs(2 * y / sigma ** 2)
    term = np.log1p(np.exp(-np.clip(llr, 0, 500))) / np.log(2)
    return 1.0 - np.sum(p_y * term * dy)


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range
    f_lo = compute_bpsk_capacity(lo, rate) - rate
    f_hi = compute_bpsk_capacity(hi, rate) - rate
    if f_lo * f_hi > 0:
        scan = np.linspace(lo, hi, num_points)
        vals = [compute_bpsk_capacity(eb, rate) - rate for eb in scan]
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] <= 0:
                return brentq(lambda x: compute_bpsk_capacity(x, rate) - rate, scan[i], scan[i + 1])
        return float(scan[np.argmin(np.abs(vals))])
    return brentq(lambda x: compute_bpsk_capacity(x, rate) - rate, lo, hi)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-6) for r in results]
        ax.semilogy(eb, bler, 'o-', linewidth=1.5, markersize=4, label=label)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_or_rate, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for N in N_list:
            if K_or_rate is None:
                K = N // 2
            elif isinstance(K_or_rate, float):
                K = int(N * K_or_rate)
            else:
                K = K_or_rate
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db, rate=rate)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=100) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=100) + '\n')
            f.write('-' * 53 + '\n')
