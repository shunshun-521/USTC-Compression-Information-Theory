"""
仅运行实验二/三未完成部分（加速版）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs('results', exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
MAX_FRAMES = 50000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(1.0, 10.5, 0.5)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}

# 加载已有 SC / SCL L=2 / L=4 结果
for label, path in [
    ('SC (L=1)', 'results/exp2_sc_N512_R0.5.csv'),
    ('SCL (L=2)', 'results/exp2_scl_L2_N512_R0.5.csv'),
    ('SCL (L=4)', 'results/exp2_scl_L4_N512_R0.5.csv'),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

for L in [8]:
    print(f'SCL L={L}')

    def scl_decoder(llr_ch, _L=L):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, 'scl',
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_results[f'SCL (L={L})'] = results
    save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

print('CA-SCL L=8')

def cascl_decoder(llr_ch):
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
    return u_hat, None

results_cascl = run_simulation(
    N, K, EB_N0_RANGE, cascl_decoder, 'scl',
    MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
    info_indices=info_idx, verbose=True,
)
all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = results_cascl
save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')
save_results_csv(results_cascl, 'results/exp2_scl_N512_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results, f'SCL vs SC BLER (N={N}, R={RATE})',
    'results/fig2_scl_bler.png', shannon_limit_db=shannon_db,
)

labels = list(all_results.keys())
avg_times = [np.mean([r['avg_decode_time'] for r in v]) * 1000 for v in all_results.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, avg_times)
ax.set_xlabel('Decoder')
ax.set_ylabel('Avg Decode Time (ms)')
ax.set_title(f'Decoding Time vs List Size (N={N})')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('results/fig2_decode_time.png', dpi=150)
plt.savefig('results/fig2_decode_time.pdf')
plt.close()

# 实验三 N=512 BP
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
all_results3 = {}

for label, path in [
    ('SC', 'results/exp3_sc_N512_R0.5.csv'),
    ('SCL (L=4)', 'results/exp3_scl_N512_R0.5.csv'),
]:
    if os.path.exists(path):
        all_results3[label] = load_results_csv(path)

bp_decoder = BPDecoder(N, frozen_bits, max_iter=30)

def bp_d(llr_ch):
    u_hat, num_iters = bp_decoder.decode(llr_ch)
    return u_hat, num_iters

print('BP N=512')
r_bp = run_simulation(
    N, K, EB_N0_RANGE, bp_d, 'bp', MAX_FRAMES, MIN_ERRORS,
    info_indices=info_idx, verbose=True,
)
all_results3['BP (max_iter=30)'] = r_bp
save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results3, f'SC vs SCL vs BP (N={N}, R={RATE})',
    f'results/fig3_bp_N{N}_bler.png', shannon_limit_db=shannon_db,
)

eb_n0_vals = [r['eb_n0_db'] for r in r_bp]
avg_iters = [r['avg_iters'] for r in r_bp]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eb_n0_vals, avg_iters, 'o-', color='purple')
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Avg Iterations')
ax.set_title(f'BP Average Iterations (N={N}, max_iter=30)')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
plt.close()

print('补充实验完成。')
