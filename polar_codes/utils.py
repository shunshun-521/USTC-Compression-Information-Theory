"""工具函数：结果保存、绘图、容量计算"""
import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.2f}",
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果。"""
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
    计算 BPSK-AWGN 信道互信息（bits/channel use）。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        sigma = 1.0 / np.sqrt(2 * rate * (10 ** (eb_n0_db / 10.0)))
        y_grid = np.linspace(-8 * sigma, 8 * sigma, 2000)
        dy = y_grid[1] - y_grid[0]
        p_y = 0.5 * (np.exp(-((y_grid - 1) ** 2) / (2 * sigma ** 2))
                     + np.exp(-((y_grid + 1) ** 2) / (2 * sigma ** 2)))
        p_y /= np.sum(p_y) * dy
        h_cond = 0.0
        for x_val in (-1.0, 1.0):
            p_yx = np.exp(-((y_grid - x_val) ** 2) / (2 * sigma ** 2))
            p_yx /= np.sum(p_yx) * dy
            p_xg = 0.5 * p_yx / np.maximum(p_y, 1e-300)
            mask = p_xg > 1e-15
            h_cond -= np.sum(p_xg[mask] * np.log2(p_xg[mask]) * p_y[mask]) * dy
        capacities.append(max(0.0, min(1.0, 1.0 - h_cond)))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = compute_bpsk_capacity(eb_grid, rate)
    idx = np.argmin(np.abs(caps - rate))
    lo, hi = eb_grid[max(0, idx - 1)], eb_grid[min(num_points - 1, idx + 1)]
    for _ in range(50):
        mid = (lo + hi) / 2
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, marker=markers[i % len(markers)], label=label, linewidth=1.5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
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
            f.write(np.array2string(info_idx, max_line_width=80) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=80) + '\n')
            f.write('-' * 53 + '\n')
