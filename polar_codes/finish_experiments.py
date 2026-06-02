"""
补全/重绘实验结果：从已有 CSV 加载并生成缺失图像，或补跑 SCL L=8 / CA-SCL。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import MAX_FRAMES, MIN_ERRORS, QUICK


def main():
    from utils import load_results_csv, plot_bler_curves, find_capacity_limit

    os.makedirs('results', exist_ok=True)
    rate = 0.5

    # 实验一：合并 CSV 绘图
    exp1 = {}
    for N in ([256, 512, 1024] if not QUICK else [256, 512]):
        path = f'results/exp1_sc_N{N}_R0.5.csv'
        if os.path.exists(path):
            exp1[f'SC, N={N}, K={N // 2}'] = load_results_csv(path)
    if exp1:
        plot_bler_curves(
            exp1,
            f'SC Decoder BLER vs Eb/N0 (R={rate})',
            'results/fig1_sc_bler.png',
            find_capacity_limit(rate),
        )
        print('fig1_sc_bler 已更新')

    # 实验二
    exp2 = {}
    for key, path in [
        ('SC (L=1)', 'results/exp2_sc_N512_R0.5.csv'),
        ('SCL (L=2)', 'results/exp2_scl_L2_N512_R0.5.csv'),
        ('SCL (L=4)', 'results/exp2_scl_L4_N512_R0.5.csv'),
        ('SCL (L=8)', 'results/exp2_scl_L8_N512_R0.5.csv'),
        ('CA-SCL', 'results/exp2_cascl_L8_N512_R0.5.csv'),
        ('CA-SCL', 'results/exp2_scl_N512_R0.5.csv'),
    ]:
        if os.path.exists(path) and key not in exp2:
            exp2[key] = load_results_csv(path)
    if len(exp2) >= 2:
        plot_bler_curves(
            exp2,
            f'SCL vs SC BLER (N=512, R={rate})',
            'results/fig2_scl_bler.png',
            find_capacity_limit(rate),
        )
        print('fig2_scl_bler 已更新')

    # 实验三
    for N in [256, 512]:
        exp3 = {}
        for dec, suffix in [
            ('SC', 'sc'), ('SCL', 'scl'), ('BP', 'bp'),
        ]:
            path = f'results/exp3_{suffix}_N{N}_R0.5.csv'
            if os.path.exists(path):
                exp3[dec if dec != 'SCL' else 'SCL (L=4)'] = load_results_csv(path)
        if len(exp3) >= 2:
            plot_bler_curves(
                exp3,
                f'SC vs SCL vs BP (N={N}, R={rate})',
                f'results/fig3_bp_N{N}_bler.png',
                find_capacity_limit(rate),
            )
            print(f'fig3_bp_N{N}_bler 已更新')

    # 可选：补跑 exp2 SCL L=8（需 POLAR_FINISH_L8=1，耗时很长）
    if (
        not QUICK
        and os.environ.get('POLAR_FINISH_L8', '0') == '1'
        and not os.path.exists('results/exp2_scl_L8_N512_R0.5.csv')
    ):
        import numpy as np
        from construction import ga_construction
        from decoder_scl import SCLDecoder
        from simulation import run_simulation
        from utils import save_results_csv

        N, K = 512, 256
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        eb = np.arange(1.0, 5.5, 0.25)

        def dec8(llr):
            u, _ = SCLDecoder(N, frozen_bits, list_size=8).decode(llr)
            return u, None

        print('补跑 SCL L=8 ...')
        res = run_simulation(
            N, K, eb, dec8, 'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        save_results_csv(res, 'results/exp2_scl_L8_N512_R0.5.csv')


if __name__ == '__main__':
    main()
