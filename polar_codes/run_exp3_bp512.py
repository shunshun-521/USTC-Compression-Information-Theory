"""补全实验三 N=512 BP 仿真（独立运行）"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs('results', exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 20000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
fb = frozen_bits.astype(bool)

all_results = {}
for label, path in [
    ('SC', 'results/exp3_sc_N512_R0.5.csv'),
    ('SCL (L=4)', 'results/exp3_scl_N512_R0.5.csv'),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

bp_decoder = BPDecoder(N, fb, max_iter=MAX_ITER)


def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters


print('Running BP N=512...')
r_bp = run_simulation(
    N, K, EB_N0_RANGE, bp_d, 'bp', MAX_FRAMES, MIN_ERRORS,
    info_indices=info_idx, verbose=True,
)
all_results[f'BP (max_iter={MAX_ITER})'] = r_bp
save_results_csv(r_bp, 'results/exp3_bp_N512_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results, f'SC vs SCL vs BP (N={N}, R={RATE})',
    'results/fig3_bp_N512_bler.png', shannon_limit_db=shannon_db,
)

eb_n0_vals = [r['eb_n0_db'] for r in r_bp]
avg_iters = [r['avg_iters'] for r in r_bp]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eb_n0_vals, avg_iters, 'o-', color='purple')
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Avg Iterations')
ax.set_title(f'BP Average Iterations (N={N}, max_iter={MAX_ITER})')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/fig3_bp_N512_iters.png', dpi=150)
plt.savefig('results/fig3_bp_N512_iters.pdf')
plt.close()
print('Done.')
