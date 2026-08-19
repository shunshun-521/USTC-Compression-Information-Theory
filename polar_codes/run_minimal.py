"""生成必需结果文件（缩减仿真量）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs('results', exist_ok=True)

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit
import matplotlib.pyplot as plt

RATE = 0.5
DESIGN = 2.5
EB = np.arange(2.0, 8.0, 1.0)
MAX_F, MIN_E = 2000, 30

# exp1 N=1024 if missing
for N in [1024]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int); fb[info_idx] = 0
    r = run_simulation(N, K, EB, lambda llr, f=fb: (sc_decode(llr, f), None),
                       'sc', MAX_F, MIN_E, info_indices=info_idx, verbose=True)
    save_results_csv(r, f'results/exp1_sc_N{N}_R0.5.csv')

# exp2
N = 512; K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN)
fb = np.ones(N, dtype=int); fb[info_idx] = 0
scl = SCLDecoder(N, fb, list_size=4)
r = run_simulation(N, K, EB, lambda llr, d=scl: (d.decode(llr)[0], None),
                   'scl', MAX_F, MIN_E, info_indices=info_idx, verbose=True)
save_results_csv(r, 'results/exp2_scl_N512_R0.5.csv')

# exp3
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int); fb[info_idx] = 0
    bp = BPDecoder(N, fb, max_iter=50)
    r = run_simulation(N, K, EB, lambda llr, d=bp: d.decode(llr),
                       'bp', MAX_F, MIN_E, info_indices=info_idx, verbose=True)
    save_results_csv(r, f'results/exp3_bp_N{N}_R0.5.csv')

print("minimal results done")
