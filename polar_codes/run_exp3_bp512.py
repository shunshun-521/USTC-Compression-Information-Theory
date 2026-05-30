"""补跑实验三 N=512 的 BP 仿真（SC/SCL 已完成）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

N = 512
K = N // 2
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 3000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)
results_bp = run_simulation(
    N, K, EB_N0_RANGE,
    lambda llr: bp_decoder.decode(llr),
    'bp', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
)
save_results_csv(results_bp, f'results/exp3_bp_N{N}_R0.5.csv')

all_results = {
    'SC': load_results_csv(f'results/exp3_sc_N{N}_R0.5.csv'),
    'SCL (L=4)': load_results_csv(f'results/exp3_scl_N{N}_R0.5.csv'),
    f'BP (max_iter={MAX_ITER})': results_bp,
}
shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f'SC vs SCL vs BP (N={N}, R={RATE})',
    f'results/fig3_bp_N{N}_bler.png',
    shannon_limit_db=shannon_db,
)

if plt is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['eb_n0_db'] for r in results_bp], [r['avg_iters'] for r in results_bp], 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Average Iterations (N={N}, max_iter={MAX_ITER})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('N=512 BP 补跑完成。')
