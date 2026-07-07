"""工具函数：结果保存、绘图、容量计算"""
import os
import csv
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
            avg_iters = r['avg_iters'] if r['avg_iters'] is not None else ''
            writer.writerow([
                f"{r['eb_n0_db']:.2f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                avg_iters if avg_iters != '' else '',
            ])


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果"""
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
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') else None,
            })
    return results


def _log2_one_plus_exp(x):
    """数值稳定的 log2(1 + exp(x))"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = x[pos] / np.log(2) + np.log2(1 + np.exp(-x[pos]))
    out[~pos] = np.log2(1 + np.exp(x[~pos]))
    return out


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    C = 1 - E[log2(1 + exp(-|LLR(y)|))]，y 为 BPSK-AWGN 接收符号。
    """
    capacities = []
    for eb_n0_db in eb_n0_db_list:
        sigma = 1.0 / np.sqrt(2 * rate * (10 ** (eb_n0_db / 10)))
        var = sigma ** 2
        norm = 1.0 / np.sqrt(2 * np.pi * var)

        def integrand(y):
            llr = 2.0 * y / var
            pdf = (
                0.5 * norm * np.exp(-(y - 1) ** 2 / (2 * var))
                + 0.5 * norm * np.exp(-(y + 1) ** 2 / (2 * var))
            )
            return _log2_one_plus_exp(-np.abs(llr)) * pdf

        val, _ = integrate.quad(integrand, -np.inf, np.inf, limit=200)
        capacities.append(1 - val)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-10, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    lo, hi = eb_n0_range[0], eb_n0_range[1]
    cap_lo = compute_bpsk_capacity([lo], rate)[0]
    cap_hi = compute_bpsk_capacity([hi], rate)[0]
    if cap_lo > rate:
        return float(lo)
    if cap_hi < rate:
        return float(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        cap = compute_bpsk_capacity([mid], rate)[0]
        if cap < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, marker=markers[i % len(markers)],
                    linestyle='-', label=label, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位集合和冻结位集合保存到文本文件"""
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
