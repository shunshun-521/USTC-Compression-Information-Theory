"""补跑实验三 N=512 的 SC/SCL 仿真。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv

N = 512
K = N // 2
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = __import__("numpy").arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0, RATE)
frozen_bits = __import__("numpy").ones(N, dtype=int)
frozen_bits[info_idx] = 0

def sc_d(llr):
    return sc_decode(llr, frozen_bits), None

print("SC N=512")
save_results_csv(
    run_simulation(N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx),
    "results/exp3_sc_N512_R0.5.csv",
)

def scl_d(llr):
    u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
    return u, None

print("SCL N=512")
save_results_csv(
    run_simulation(N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx),
    "results/exp3_scl_N512_R0.5.csv",
)

print("Done")
