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


def _bpsk_capacity(snr_linear):
    """BPSK 离散输入信道容量（bits/channel use）。"""

    def integrand(y):
        t = -2.0 * snr_linear * y ** 2
        phi = np.exp(-y ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
        if t > 30:
            log2_term = t / np.log(2.0)
        elif t < -30:
            log2_term = 0.0
        else:
            log2_term = np.log2(1.0 + np.exp(t))
        return phi * log2_term

    val, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    return 1.0 - val


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
        capacities.append(_bpsk_capacity(snr))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 6), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_n0_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_n0_vals, rate)

    for i in range(len(caps) - 1):
        if caps[i] <= rate <= caps[i + 1] or caps[i] >= rate >= caps[i + 1]:
            denom = caps[i + 1] - caps[i]
            if abs(denom) < 1e-12:
                return eb_n0_vals[i]
            t = (rate - caps[i]) / denom
            return eb_n0_vals[i] + t * (eb_n0_vals[i + 1] - eb_n0_vals[i])

    if caps[-1] < rate:
        return eb_n0_range[1]
    return eb_n0_range[0]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for idx, (label, results) in enumerate(results_dict.items()):
        eb_n0 = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-6) for r in results]
        ax.semilogy(
            eb_n0, bler,
            marker=markers[idx % len(markers)],
            label=label,
            linewidth=1.5,
            markersize=5,
        )

    if shannon_limit_db is not None:
        ax.axvline(x=shannon_limit_db, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)

            f.write('=' * 53 + '\n')
            f.write(
                f'N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n'
            )
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
