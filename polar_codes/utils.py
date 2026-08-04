"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    """
    将仿真结果保存为 CSV 文件。
    列：eb_n0_db, bler, ber, num_errors, num_frames, avg_decode_time_ms, avg_iters
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fieldnames = [
        'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
        'avg_decode_time_ms', 'avg_iters',
    ]
    with open(filepath, 'w', newline='') as f:
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
                'avg_iters': r['avg_iters'] if r['avg_iters'] is not None else '',
            })


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
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000,
                'avg_iters': float(row['avg_iters']) if row['avg_iters'] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    capacities = []
    xs = np.linspace(0, 12, 10000)
    for eb_n0_db in eb_n0_db_list:
        gamma = 10 ** (eb_n0_db / 10.0)
        integrand = np.exp(-xs ** 2) * np.log2(1 + np.exp(-gamma * xs ** 2))
        capacities.append(1 - (2 / np.sqrt(np.pi)) * np.trapezoid(integrand, xs))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=2000):
    """
    找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。
    """
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_n0_vals, rate)

    for i in range(1, len(eb_n0_vals)):
        if (caps[i - 1] - rate) * (caps[i] - rate) <= 0:
            e0, e1 = eb_n0_vals[i - 1], eb_n0_vals[i]
            c0, c1 = caps[i - 1], caps[i]
            if c1 != c0:
                return float(e0 + (rate - c0) * (e1 - e0) / (c1 - c0))

    idx = np.argmin(np.abs(caps - rate))
    return float(eb_n0_vals[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """
    绘制 BLER-Eb/N0 曲线。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, marker=markers[i % len(markers)], label=label, linewidth=1.5)

    if shannon_limit_db is not None:
        ax.axvline(x=shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """
    将各码长的信息位集合和冻结位集合保存到文本文件。
    """
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
