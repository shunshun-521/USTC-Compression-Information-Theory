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
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
            'avg_decode_time_ms', 'avg_iters'
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
    """从 CSV 文件加载仿真结果"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
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
    C = 1 - ∫ log2(1 + exp(-SNR * y^2)) φ(y) dy, SNR = 2R * 10^{Eb/N0/10}
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))

        def integrand(y):
            val = -snr * y * y
            if val < -700:
                log_term = -val / np.log(2)
            elif val > 700:
                log_term = 0.0
            else:
                log_term = np.log2(1.0 + np.exp(val))
            return log_term * np.exp(-0.5 * y ** 2) / np.sqrt(2.0 * np.pi)

        integral, _ = integrate.quad(integrand, -20.0, 20.0, limit=200)
        capacities.append(1.0 - integral)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 15), num_points=500):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    for i in range(len(eb_grid) - 1):
        if (caps[i] - rate) * (caps[i + 1] - rate) <= 0:
            t = (rate - caps[i]) / (caps[i + 1] - caps[i] + 1e-15)
            return eb_grid[i] + t * (eb_grid[i + 1] - eb_grid[i])
    return eb_grid[np.argmin(np.abs(caps - rate))]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Capacity limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path, rate=0.5):
    """将各码长的信息位/冻结位集合保存到文本文件"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for N in N_list:
            k = K if K is not None else int(N * rate)
            info_idx, frozen_idx, _ = ga_construction(N, k, design_eb_n0_db, rate=rate)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={k}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
