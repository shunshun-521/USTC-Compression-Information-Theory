"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fieldnames = [
        'eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
        'avg_decode_time', 'avg_iters',
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            row['avg_decode_time'] = r['avg_decode_time'] * 1000
            if row['avg_iters'] is None:
                row['avg_iters'] = ''
            writer.writerow(row)


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
    results = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'eb_n0_db': float(row['eb_n0_db']),
                'bler': float(row['bler']),
                'ber': float(row['ber']),
                'num_errors': int(row['num_errors']),
                'num_frames': int(row['num_frames']),
                'avg_decode_time': float(row['avg_decode_time']) / 1000,
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') else None,
            })
    return results


def _bpsk_capacity_integrand(y, snr):
    return np.log2(1 + np.exp(-snr * y)) * np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb / 10))
        cap, _ = integrate.quad(lambda y: _bpsk_capacity_integrand(y, snr), -np.inf, np.inf)
        caps.append(1 - cap)
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    if idx == 0 or idx == len(eb_grid) - 1:
        for eb in np.linspace(-5, 25, 5000):
            cap = compute_bpsk_capacity([eb], rate)[0]
            if cap >= rate:
                return eb
    return eb_grid[idx]


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-6) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, markersize=4)
    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', label=f'Shannon limit ({shannon_limit_db:.2f} dB)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合。"""
    from construction import ga_construction
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K_n = N // 2 if K is None else K
            info_idx, frozen_idx, _ = ga_construction(N, K_n, design_eb_n0_db)
            rate = K_n / N
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_n}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {K_n}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {N - K_n}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
