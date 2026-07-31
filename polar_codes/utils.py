"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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
            writer.writerow([
                r['eb_n0_db'],
                f"{r['bler']:.6e}",
                f"{r['ber']:.6e}",
                r['num_errors'],
                r['num_frames'],
                f"{r['avg_decode_time'] * 1000:.6f}",
                '' if r['avg_iters'] is None else f"{r['avg_iters']:.2f}",
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
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') and row['avg_iters'] else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 信道容量（bits/channel use）。
    使用标准积分公式：C = (1/pi) * ∫_0^{pi/2} log2(1 + Es/N0 * sin^2(theta)) d(theta)
    其中 Es/N0 = Eb/N0 * R（线性）
    """
    from scipy import integrate

    capacities = []
    for eb_n0_db in eb_n0_db_list:
        es_n0 = 10 ** (eb_n0_db / 10.0) * rate

        def integrand(theta):
            return np.log(1.0 + es_n0 * np.sin(theta) ** 2) / np.pi

        cap, _ = integrate.quad(integrand, 0, np.pi / 2)
        capacities.append(cap / np.log(2))
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 15), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    from scipy.optimize import brentq

    def objective(eb_n0_db):
        cap = compute_bpsk_capacity([eb_n0_db], rate)[0]
        return cap - rate

    try:
        return brentq(objective, eb_n0_range[0], eb_n0_range[1])
    except ValueError:
        eb_grid = np.linspace(eb_n0_range[0], eb_n0_range[1], num_points)
        caps = compute_bpsk_capacity(eb_grid, rate)
        idx = np.searchsorted(caps, rate)
        if idx == 0:
            return eb_grid[0]
        if idx >= len(eb_grid):
            return eb_grid[-1]
        c0, c1 = caps[idx - 1], caps[idx]
        e0, e1 = eb_grid[idx - 1], eb_grid[idx]
        if c1 == c0:
            return e0
        return e0 + (rate - c0) * (e1 - e0) / (c1 - c0)


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in results_dict.items():
        eb = [r['eb_n0_db'] for r in results]
        bler = [max(r['bler'], 1e-7) for r in results]
        ax.semilogy(eb, bler, 'o-', label=label, markersize=4)

    if shannon_limit_db is not None:
        ax.axvline(shannon_limit_db, color='gray', linestyle='--', label=f'Shannon limit ({shannon_limit_db:.2f} dB)')

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
    """保存信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            if K is None:
                K_n = N // 2
            else:
                K_n = K
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
