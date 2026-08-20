"""完成实验三 N=512 BP 仿真（补跑，加速参数）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, load_results_csv, find_capacity_limit

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

os.makedirs('results', exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
MAX_ITER = 20
MAX_FRAMES = 2000
MIN_ERRORS = 20
EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)


def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters


print('BP N=512 (accelerated)')
r_bp = run_simulation(
    N, K, EB_N0_RANGE, bp_d, 'bp', MAX_FRAMES, MIN_ERRORS,
    info_indices=info_idx, verbose=True,
)
save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

r_sc = load_results_csv('results/exp3_sc_N512_R0.5.csv')
r_scl = load_results_csv('results/exp3_scl_N512_R0.5.csv')
all_results = {
    'SC': r_sc,
    'SCL (L=4)': r_scl,
    f'BP (max_iter={MAX_ITER})': r_bp,
}
shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results, f'SC vs SCL vs BP (N={N}, R={RATE})',
    f'results/fig3_bp_N{N}_bler.png', shannon_limit_db=shannon_db,
)

if HAS_MPL:
    eb_n0_vals = [r['eb_n0_db'] for r in r_bp]
    avg_iters = [r['avg_iters'] for r in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb_n0_vals, avg_iters, 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Average Iterations (N={N}, max_iter={MAX_ITER})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('N=512 BP 补跑完成')
