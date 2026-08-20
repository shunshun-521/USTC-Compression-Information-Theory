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
                r['bler'],
                r['ber'],
                r['num_errors'],
                r['num_frames'],
                r['avg_decode_time'] * 1000,
                r['avg_iters'] if r['avg_iters'] is not None else '',
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
                'avg_decode_time': float(row['avg_decode_time_ms']) / 1000,
                'avg_iters': float(row['avg_iters']) if row.get('avg_iters') else None,
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    C = 1 - E_y[log2(1 + exp(-2*s*y))], s = SNR = 2R * 10^{Eb/N0/10}
    """
    from scipy import integrate

    capacities = []
    for eb_n0_db in eb_n0_db_list:
        snr = 2 * rate * (10 ** (eb_n0_db / 10))
        s = snr

        def integrand(y):
            val = -2 * s * y
            if val < -50:
                log_term = 0.0
            elif val > 50:
                log_term = val / np.log(2)
            else:
                log_term = np.log2(1 + np.exp(val))
            return log_term * np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)

        integral, _ = integrate.quad(integrand, -10, 10)
        capacities.append(1 - integral)
    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-5, 20), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）"""
    try:
        from scipy import integrate
    except ImportError:
        # 无 scipy 时使用近似
        return 0.0

    lo, hi = eb_n0_range
    eb_vals = np.linspace(lo, hi, num_points)

    def capacity_at(eb_db):
        snr = 2 * rate * (10 ** (eb_db / 10))
        s = snr

        def integrand(y):
            val = -2 * s * y
            if val < -50:
                log_term = 0.0
            elif val > 50:
                log_term = val / np.log(2)
            else:
                log_term = np.log2(1 + np.exp(val))
            return log_term * np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)

        integral, _ = integrate.quad(integrand, -10, 10)
        return 1 - integral

    caps = [capacity_at(e) for e in eb_vals]
    idx = np.argmin(np.abs(np.array(caps) - rate))
    return float(eb_vals[idx])


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线"""
    if not HAS_MPL:
        return

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
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path)
    plt.close()


def save_frozen_set_info(N_list, K_override, design_eb_n0_db, save_path):
    """保存各码长的信息位/冻结位集合"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        for N in N_list:
            K = K_override if K_override is not None else N // 2
            rate = K / N
            info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db)
            f.write('=' * 53 + '\n')
            f.write(f'N={N}, K={K}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n')
            f.write('=' * 53 + '\n')
            f.write(f'Info indices (all {len(info_idx)}):\n')
            f.write(np.array2string(info_idx, threshold=N) + '\n')
            f.write(f'Frozen indices (all {len(frozen_idx)}):\n')
            f.write(np.array2string(frozen_idx, threshold=N) + '\n')
            f.write('-' * 53 + '\n')
