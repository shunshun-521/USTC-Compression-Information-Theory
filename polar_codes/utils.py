"""工具函数：结果保存、绘图、容量计算"""
import csv
import os

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
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
                'avg_decode_time_ms': r['avg_decode_time'] * 1000.0,
                'avg_iters': '' if r['avg_iters'] is None else r['avg_iters'],
            })


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表"""
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
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000.0,
                'avg_iters': None if row['avg_iters'] == '' else float(row['avg_iters']),
            })
    return results


def bpsk_capacity(snr_linear):
    """
    BPSK 离散输入信道容量（bits/channel use）。
    snr_linear = Es/N0 = 2 * R * 10^(Eb/N0/10)
    """
    sigma2 = 1.0 / (2.0 * snr_linear)
    sigma = np.sqrt(sigma2)

    def entropy_y():
        def integrand(y):
            p0 = 0.5 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-(y - 1.0) ** 2 / (2.0 * sigma2))
            p1 = 0.5 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-(y + 1.0) ** 2 / (2.0 * sigma2))
            p = p0 + p1
            if p < 1e-300:
                return 0.0
            return -p * np.log2(p)

        h, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
        return h

    h_y = entropy_y()
    h_y_given_x = 0.5 * np.log2(2.0 * np.pi * np.e * sigma2)
    return h_y - h_y_given_x


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """计算给定 Eb/N0 列表下的 BPSK 信道容量。"""
    caps = []
    for eb in eb_n0_db_list:
        snr = 2.0 * rate * (10.0 ** (eb / 10.0))
        caps.append(bpsk_capacity(snr))
    return np.array(caps)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range
    eb_vals = np.linspace(lo, hi, num_points)
    caps = compute_bpsk_capacity(eb_vals, rate)
    diff = caps - rate
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return float(eb_vals[np.argmin(np.abs(diff))])
    i = idx[0]
    x0, x1 = eb_vals[i], eb_vals[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
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

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件。"""
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
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
