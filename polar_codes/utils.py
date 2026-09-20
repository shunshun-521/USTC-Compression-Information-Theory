"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
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
    """从 CSV 文件加载仿真结果"""
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


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    采用数值稳定的 |y| 形式近似：
    C = 1 - E_y[log2(1 + exp(-2*s*|y|))], s = 2R * 10^{Eb/N0/10}
    """
    eb_n0_db_list = np.atleast_1d(np.asarray(eb_n0_db_list, dtype=np.float64))
    log2 = np.log(2.0)
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        s = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))

        def integrand(y):
            x = np.clip(-2.0 * s * np.abs(y), -700.0, 700.0)
            return (np.logaddexp(0.0, x) / log2) * np.exp(-0.5 * y ** 2)

        val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
        capacities.append(float(1.0 - val / np.sqrt(2.0 * np.pi)))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-3, 8), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""

    def capacity_minus_rate(eb_n0_db):
        return compute_bpsk_capacity([eb_n0_db], rate)[0] - rate

    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    for i in range(len(grid) - 1):
        f_lo = caps[i] - rate
        f_hi = caps[i + 1] - rate
        if f_lo * f_hi <= 0 and np.isfinite(f_lo) and np.isfinite(f_hi):
            return float(brentq(capacity_minus_rate, grid[i], grid[i + 1]))

    idx = np.argmin(np.abs(caps - rate))
    return float(grid[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            rate = K_n / N
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db, rate)
            f.write('=' * 53 + '\n')
            f.write(
                f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n'
            )
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, separator=' ', max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, separator=' ', max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
