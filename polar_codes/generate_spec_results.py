"""
生成规范要求的结果文件（N=256/512，中等帧数）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("results", exist_ok=True)

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

DESIGN_EBN0 = 2.5
MAX_FRAMES = 10000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(0.0, 5.5, 0.5)

save_frozen_set_info([256, 512], None, DESIGN_EBN0, "results/frozen_sets.txt")

# exp1: SC N=256, 512
all_sc = {}
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, fb=frozen_bits):
        return sc_decode(llr_ch, fb), None

    results = run_simulation(
        N, K, EB_N0_RANGE, decoder, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, frozen_bits=frozen_bits,
    )
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")
    all_sc[f"SC, N={N}"] = results

plot_bler_curves(
    all_sc, "SC Decoder BLER", "results/fig1_sc_bler.png",
    shannon_limit_db=find_capacity_limit(0.5),
)

# exp2: SCL N=512
N, K = 512, 256
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
scl = SCLDecoder(N, frozen_bits, list_size=4)


def scl_decoder(llr_ch):
    return scl.decode(llr_ch)[0], None


results_scl = run_simulation(
    N, K, np.arange(1.0, 5.5, 0.5), scl_decoder, "scl",
    MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=frozen_bits,
)
save_results_csv(results_scl, "results/exp2_scl_N512_R0.5.csv")

# exp3: BP N=256, 512
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits)

    def bp_decoder(llr_ch, _bp=bp):
        return _bp.decode(llr_ch)

    results = run_simulation(
        N, K, np.arange(1.0, 5.5, 0.5), bp_decoder, "bp",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=frozen_bits,
    )
    save_results_csv(results, f"results/exp3_bp_N{N}_R0.5.csv")

print("规范结果文件已生成。")
