"""生成规范参数下的仿真结果（N=256/512，5000 帧，30 最少错误）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs("results", exist_ok=True)

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 5000
MIN_ERRORS = 30
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

# SC N=256, 512
all_sc = {}
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr):
        return sc_decode(llr, frozen_bits), None

    print(f"SC N={N}")
    results = run_simulation(
        N, K, EB_N0_RANGE, decoder, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")
    all_sc[f"SC N={N}"] = results

plot_bler_curves(all_sc, "SC BLER (spec)", "results/fig1_sc_bler.png",
                 find_capacity_limit(RATE))

# SCL N=512
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

print("SCL N=512 L=4")
scl_results = run_simulation(
    N, K, EB_N0_RANGE,
    lambda llr: SCLDecoder(N, frozen_bits, 4).decode(llr) + (None,),
    "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
)
save_results_csv(scl_results, "results/exp2_scl_N512_R0.5.csv")

# BP N=256, 512
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits)

    print(f"BP N={N}")
    results = run_simulation(
        N, K, EB_N0_RANGE,
        lambda llr, _bp=bp: _bp.decode(llr),
        "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, f"results/exp3_bp_N{N}_R0.5.csv")

print("规范结果生成完成。")
