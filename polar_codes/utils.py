"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fieldnames = [
        'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
        'avg_decode_time_ms', 'avg_iters',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'eb_n0_db': r['eb_n0_db'],
                'bler': r['bler'],
                'ber': r['ber'],
                'num_errors': r['num_errors'],
                'num_frames': r['num_frames'],
                'avg_decode_time_ms': r['avg_decode_time'] * 1000,
                'avg_iters': '' if r['avg_iters'] is None else r['avg_iters'],
            })


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_iters = row['avg_iters'].strip()
            results.append({
                'eb_n0_db': float(row['eb_n0_db']),
                'bler': float(row['bler']),
                'ber': float(row['ber']),
                'num_errors': int(row['num_errors']),
                'num_frames': int(row['num_frames']),
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000,
                'avg_iters': None if avg_iters == '' else float(avg_iters),
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    snr = 2 * rate * (10 ** (eb_n0_db / 10))

    def integrand(y):
        return np.log2(1 + np.exp(-2 * snr * y)) * np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)

    val, _ = integrate.quad(integrand, -20, 20, limit=200)
    return 1 - val


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([compute_bpsk_capacity(eb, rate) for eb in eb_grid])
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        return float(eb_grid[idx])
    x0, x1 = eb_grid[idx - 1], eb_grid[idx + 1]
    c0 = compute_bpsk_capacity(x0, rate) - rate
    c1 = compute_bpsk_capacity(x1, rate) - rate
    for _ in range(50):
        xm = (x0 + x1) / 2
        cm = compute_bpsk_capacity(xm, rate) - rate
        if c0 * cm <= 0:
            x1, c1 = xm, cm
        else:
            x0, c0 = xm, cm
    return (x0 + x1) / 2


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-6) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, threshold=N, separator=' ') + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, threshold=N, separator=' ') + '\n')
            f.write('-' * 53 + '\n')
