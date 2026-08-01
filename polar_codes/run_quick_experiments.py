"""快速实验运行器（缩减参数，用于验证完整流程）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, verify_sc_decoder
from decoder_scl import SCLDecoder, verify_scl_equals_sc
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs("results", exist_ok=True)

u = np.array([1, 0, 1, 1])
assert np.array_equal(polar_encode(u), [1, 1, 0, 1])
verify_sc_decoder(N=64, K=32, num_frames=30, eb_n0_db=10.0)
verify_scl_equals_sc(N=64, K=32, num_frames=10)
print("单元测试通过")

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 15000
MIN_ERRORS = 50
EB_N0 = np.arange(1.0, 5.0, 0.5)

save_frozen_set_info([256, 512], None, DESIGN_EBN0, "results/frozen_sets.txt", rate=RATE)

all_sc = {}
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    fb = np.ones(N, dtype=bool)
    fb[info_idx] = False

    def sc_dec(llr, _fb=fb):
        return sc_decode(llr, _fb), None

    r = run_simulation(
        N, K, EB_N0, sc_dec, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, frozen_bits=fb,
    )
    all_sc[f"SC, N={N}"] = r
    save_results_csv(r, f"results/exp1_sc_N{N}_R0.5.csv")

plot_bler_curves(all_sc, "SC BLER", "results/fig1_sc_bler.png", find_capacity_limit(RATE))

N, K = 512, 256
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
fb512 = np.ones(N, dtype=bool)
fb512[info_idx] = False
all_scl = {}

all_scl["SC (L=1)"] = run_simulation(
    N, K, EB_N0, lambda l: (sc_decode(l, fb512), None), "sc",
    MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=fb512,
)
save_results_csv(all_scl["SC (L=1)"], f"results/exp2_sc_N{N}_R0.5.csv")

for L in [2, 4, 8]:
    def scl_dec(llr, _L=L):
        u_hat, _ = SCLDecoder(N, fb512, list_size=_L).decode(llr)
        return u_hat, None

    r = run_simulation(
        N, K, EB_N0, scl_dec, "scl", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, frozen_bits=fb512,
    )
    all_scl[f"SCL (L={L})"] = r
    save_results_csv(r, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

plot_bler_curves(all_scl, "SCL BLER", "results/fig2_scl_bler.png", find_capacity_limit(RATE))

N, K = 256, 128
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
fb256 = np.ones(N, dtype=bool)
fb256[info_idx] = False
bp = BPDecoder(N, fb256, max_iter=50)

all_bp = {
    "SC": run_simulation(
        N, K, EB_N0, lambda l: (sc_decode(l, fb256), None), "sc",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=fb256,
    ),
    "SCL (L=4)": run_simulation(
        N, K, EB_N0,
        lambda l: (SCLDecoder(N, fb256, 4).decode(l)[0], None), "scl",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=fb256,
    ),
    "BP (max_iter=50)": run_simulation(
        N, K, EB_N0, lambda l: bp.decode(l), "bp",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, frozen_bits=fb256,
    ),
}

save_results_csv(all_bp["SC"], f"results/exp3_sc_N{N}_R0.5.csv")
save_results_csv(all_bp["SCL (L=4)"], f"results/exp3_scl_N{N}_R0.5.csv")
save_results_csv(all_bp["BP (max_iter=50)"], f"results/exp3_bp_N{N}_R0.5.csv")
plot_bler_curves(all_bp, "BP compare", f"results/fig3_bp_N{N}_bler.png", find_capacity_limit(RATE))

print("快速实验完成")
