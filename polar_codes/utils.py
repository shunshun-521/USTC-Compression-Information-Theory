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
                f"{r['avg_iters']:.2f}" if r['avg_iters'] is not None else '',
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
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK-AWGN 信道容量（bits/channel use）。
    C = 1 - 0.5 * E[log2(1+exp(-2sy))|x=+1] - 0.5 * E[log2(1+exp(2sy))|x=-1]
    其中 s = 2R * 10^{Eb/N0/10} = 1/sigma^2
    """
    snr = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
    sigma = 1.0 / np.sqrt(snr)
    s = snr

    def integrand_pos(y):
        p = np.exp(-(y - 1) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
        return p * np.log2(1.0 + np.exp(-2.0 * s * y))

    def integrand_neg(y):
        p = np.exp(-(y + 1) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
        return p * np.log2(1.0 + np.exp(2.0 * s * y))

    i_pos, _ = integrate.quad(integrand_pos, -50, 50, limit=200)
    i_neg, _ = integrate.quad(integrand_neg, -50, 50, limit=200)
    return 1.0 - 0.5 * (i_pos + i_neg)


def find_capacity_limit(rate, eb_n0_range=(-2, 5), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    from scipy.optimize import brentq

    def diff(eb_db):
        return compute_bpsk_capacity(eb_db, rate) - rate

    try:
        return brentq(diff, eb_n0_range[0], eb_n0_range[1])
    except ValueError:
        eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
        caps = np.array([compute_bpsk_capacity(eb, rate) for eb in eb_grid])
        idx = np.argmin(np.abs(caps - rate))
        return eb_grid[idx]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)
            rate = K_val / N
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, threshold=N) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, threshold=N) + '\n')
            f.write('-' * 53 + '\n')
