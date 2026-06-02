"""补全实验二剩余 SCL / CA-SCL 仿真与绘图"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, load_results_csv, find_capacity_limit

os.makedirs('results', exist_ok=True)
N, RATE, K = 512, 0.5, 256
DESIGN_EBN0, CRC_LENGTH = 2.5, 8
MAX_FRAMES, MIN_ERRORS = 100000, 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}
if os.path.exists('results/exp2_sc_N512_R0.5.csv'):
    all_results['SC (L=1)'] = load_results_csv('results/exp2_sc_N512_R0.5.csv')
if os.path.exists('results/exp2_scl_L2_N512_R0.5.csv'):
    all_results['SCL (L=2)'] = load_results_csv('results/exp2_scl_L2_N512_R0.5.csv')

for L in [4, 8]:
    print(f'SCL L={L}')
    def make_dec(L_=L):
        def dec(llr):
            u, _ = SCLDecoder(N, frozen_bits, list_size=L_).decode(llr)
            return u, None
        return dec
    res = run_simulation(N, K, EB_N0_RANGE, make_dec(), 'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all_results[f'SCL (L={L})'] = res
    save_results_csv(res, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

print('CA-SCL L=8')
def cascl(llr):
    u, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr)
    return u, None
res_c = run_simulation(N, K, EB_N0_RANGE, cascl, 'scl', MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH, info_indices=info_idx)
all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = res_c
save_results_csv(res_c, f'results/exp2_cascl_L8_N{N}_R0.5.csv')
save_results_csv(res_c, f'results/exp2_scl_N{N}_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(all_results, f'SCL vs SC BLER (N={N}, R={RATE})', 'results/fig2_scl_bler.png', shannon_db)

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
print('实验二补全完成')
