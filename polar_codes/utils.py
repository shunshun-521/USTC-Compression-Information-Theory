"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


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
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.2f}",
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
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    from scipy.integrate import quad

    capacities = []
    for eb_n0_db in eb_n0_db_list:
        eb = 10.0 ** (eb_n0_db / 10.0)
        es = eb * rate

        def integrand(x):
            t = np.clip(-2.0 * es * x ** 2, -700.0, 700.0)
            return np.log2(1.0 + np.exp(t)) * np.exp(-x ** 2) / np.sqrt(np.pi)

        val, _ = quad(integrand, 0.0, np.inf, limit=200)
        capacities.append(1.0 - 2.0 * val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 12), num_points=2000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(grid, rate)
    diff = caps - rate
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    lo, hi = eb_n0_range
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, linewidth=1.5, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close(fig)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for N in N_list:
            k_val = K if K is not None else N // 2
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            rate = k_val / N
            f.write('=' * 53 + '\n')
            f.write(
                f'N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n'
            )
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {k_val}):\n')
            f.write(np.array2string(info_idx, separator=' ', max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {N - k_val}):\n')
            f.write(np.array2string(frozen_idx, separator=' ', max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
