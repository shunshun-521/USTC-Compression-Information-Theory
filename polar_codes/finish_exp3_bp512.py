"""Complete exp3 BP simulation for N=512 only."""
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
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

N = 512
K = N // 2
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 20
EB_N0_RANGE_BP512 = np.arange(3.0, 8.5, 1.0)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bool = np.ones(N, dtype=bool)
frozen_bool[info_idx] = False

all_results = {}

for label, path, decoder, dtype, max_frames in [
    ('SC', 'results/exp3_sc_N512_R0.5.csv', lambda llr: (sc_decode(llr, frozen_bool), None), 'sc', 20000),
    ('SCL (L=4)', 'results/exp3_scl_N512_R0.5.csv', lambda llr: (SCLDecoder(N, frozen_bool, 4).decode(llr)[0], None), 'scl', 20000),
]:
    if os.path.exists(path):
        from utils import load_results_csv
        all_results[label] = load_results_csv(path)
    else:
        results = run_simulation(N, K, EB_N0_RANGE, decoder, dtype, max_frames, 100, info_indices=info_idx)
        save_results_csv(results, path)
        all_results[label] = results

bp_decoder = BPDecoder(N, frozen_bool, max_iter=MAX_ITER)

def bp_d(llr):
    return bp_decoder.decode(llr)

print('Running BP for N=512...')
r_bp = run_simulation(N, K, EB_N0_RANGE_BP512, bp_d, 'bp', 150, 25, info_indices=info_idx, verbose=True)
all_results[f'BP (max_iter={MAX_ITER})'] = r_bp
save_results_csv(r_bp, 'results/exp3_bp_N512_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(all_results, f'SC vs SCL vs BP (N={N}, R={RATE})',
                 'results/fig3_bp_N512_bler.png', shannon_limit_db=shannon_db)

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
print('N=512 BP complete.')
