"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np
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
                r['eb_n0_db'],
                r['bler'],
                r['ber'],
                r['num_errors'],
                r['num_frames'],
                r['avg_decode_time'] * 1000,
                '' if r['avg_iters'] is None else r['avg_iters'],
            ])


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
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000,
                'avg_iters': (
                    None if not row['avg_iters'] else float(row['avg_iters'])
                ),
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """计算 BPSK 离散输入信道容量（bits/channel use）。"""
    eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
    snr = 2.0 * rate * eb_n0_lin

    def integrand(x):
        val = snr * x ** 2
        if val > 700:
            return 0.0
        return np.log2(1.0 + np.exp(-val))

    integral, _ = integrate.quad(integrand, 0.0, 20.0)
    return 1.0 - integral / np.pi


def find_capacity_limit(rate, eb_n0_range=(-5, 5), num_points=200):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    eb_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = [compute_bpsk_capacity(eb, rate) for eb in eb_vals]
    for i in range(len(eb_vals) - 1):
        c0, c1 = caps[i], caps[i + 1]
        if (c0 - rate) * (c1 - rate) <= 0 and c0 != c1:
            t = (rate - c0) / (c1 - c0)
            return eb_vals[i] + t * (eb_vals[i + 1] - eb_vals[i])
    return eb_vals[-1]


def plot_bler_curves(
    results_dict, title, save_path, shannon_limit_db=None,
    xlabel='Eb/N0 (dB)', ylabel='BLER',
):
    """绘制 BLER-Eb/N0 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-6) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, linewidth=2, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """保存信息位/冻结位集合。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            if K is None:
                K_val = N // 2
            else:
                K_val = K
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(
                N, K_val, design_eb_n0_db, rate=rate
            )
            f.write('=' * 53 + '\n')
            f.write(
                f'N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, '
                f'R={rate:.4f}\n'
            )
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
