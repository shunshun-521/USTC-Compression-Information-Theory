"""从已有 SC/L=2 结果继续完成实验二。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

os.makedirs('results', exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
CRC_LENGTH = 8

info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {
    'SC (L=1)': load_results_csv('results/exp2_sc_N512_R0.5.csv'),
    'SCL (L=2)': load_results_csv('results/exp2_scl_L2_N512_R0.5.csv'),
}

for L, snr, max_frames in [(4, np.array([1.0, 2.0, 3.0]), 200)]:
    print(f'\nSCL 仿真: N={N}, L={L}')

    def scl_decoder(llr_ch, _L=L):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, snr, scl_decoder, 'scl', max_frames, 15,
        info_indices=info_idx, verbose=True,
    )
    label = f'SCL (L={L})'
    all_results[label] = results
    save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')
    save_results_csv(results, f'results/exp2_scl_N{N}_R0.5.csv')

print(f'\nCA-SCL: L=8, CRC={CRC_LENGTH}')


def cascl_decoder(llr_ch):
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
    return u_hat, None


results_cascl = run_simulation(
    N, K, np.array([1.5, 2.0, 2.5, 3.0]), cascl_decoder, 'scl',
    200, 15, crc_length=CRC_LENGTH, info_indices=info_idx, verbose=True,
)
all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = results_cascl
save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f'SCL vs SC BLER (N={N}, R={RATE})',
    'results/fig2_scl_bler.png',
    shannon_limit_db=shannon_db,
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

print('\n实验二完成。')
