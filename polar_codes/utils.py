"""工具函数：结果保存、绘图、容量计算"""
import os
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from scipy import integrate
from construction import ga_construction


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eb_n0_db', 'bler', 'ber', 'num_errors', 'num_frames',
                         'avg_decode_time_ms', 'avg_iters'])
        for r in results:
            avg_iters = r.get('avg_iters')
            writer.writerow([
                f"{r['eb_n0_db']:.4f}",
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                f"{avg_iters:.2f}" if avg_iters is not None else '',
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
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters', '').strip() else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    使用标准 BPSK-AWGN 互信息数值积分。
    """
    gamma = 10 ** (eb_n0_db / 10.0)

    def integrand(t):
        z = -4.0 * gamma * t * t
        if z < -50:
            return 0.0
        if z > 50:
            return t * t * np.exp(-t * t) / np.sqrt(np.pi) * z / np.log(2)
        log_term = np.log1p(np.exp(z)) / np.log(2)
        return np.exp(-t * t) * log_term / np.sqrt(np.pi)

    val, _ = integrate.quad(integrand, 0.0, 25.0, limit=200)
    return max(0.0, 1.0 - val)


def find_capacity_limit(rate, eb_n0_range=(-30, 8), num_points=800):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    eb_vals = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
    caps = np.array([compute_bpsk_capacity(eb, rate) for eb in eb_vals])
    diff = caps - rate
    for i in range(len(diff) - 1):
        if diff[i] >= 0 and diff[i + 1] < 0:
            e0, e1 = eb_vals[i], eb_vals[i + 1]
            c0, c1 = caps[i], caps[i + 1]
            if c1 != c0:
                return e0 + (rate - c0) * (e1 - e0) / (c1 - c0)
    idx = np.argmin(np.abs(diff))
    return float(eb_vals[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, (label, results) in enumerate(results_dict.items()):
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, marker=markers[i % len(markers)],
                    label=label, linewidth=1.5, markersize=5)

    if shannon_limit_db is not None:
        ax.axvline(x=shannon_limit_db, color='gray', linestyle='--',
                   label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
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
            K_val = K if K is not None else N // 2
            rate = K_val / N
            info_idx, frozen_idx, _ = ga_construction(N, K_val, design_eb_n0_db)

            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, max_line_width=120) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, max_line_width=120) + '\n')
            f.write('-' * 53 + '\n')
