"""生成规范要求码长的仿真结果（中等帧数，用于报告核对）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv

os.makedirs("results", exist_ok=True)

EB_N0 = np.arange(1.0, 5.0, 0.5)
MAX_FRAMES = 5000
MIN_ERRORS = 50
DESIGN = 2.5

for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int)
    fb[info] = 0

    print(f"SC N={N}")
    results = run_simulation(
        N,
        K,
        EB_N0,
        lambda llr: (sc_decode(llr, fb), None),
        "sc",
        MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info,
    )
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

if True:
    N = 512
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    print(f"SCL N={N}")
    results = run_simulation(
        N,
        K,
        EB_N0,
        lambda llr, _N=N, _fb=fb: (
            SCLDecoder(_N, _fb, list_size=4).decode(llr)[0],
            None,
        ),
        "scl",
        MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info,
    )
    save_results_csv(results, f"results/exp2_scl_N512_R0.5.csv")

for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    bp = BPDecoder(N, fb, max_iter=50)
    print(f"BP N={N}")
    results = run_simulation(
        N,
        K,
        EB_N0,
        lambda llr: bp.decode(llr),
        "bp",
        MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info,
    )
    save_results_csv(results, f"results/exp3_bp_N{N}_R0.5.csv")

print("规范结果文件已生成。")
