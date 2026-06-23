"""
实验二：SCL 译码及 CRC 辅助
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

QUICK = os.environ.get("POLAR_QUICK", "0") == "1"


def _run_unit_tests():
    from tests_unit import test_encoder, test_sc_noiseless, test_scl_l1_equals_sc

    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    print("单元测试全部通过。\n")


_run_unit_tests()

N = 64 if QUICK else 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_LIST = [2, 4] if QUICK else [2, 4, 8]
MAX_FRAMES = 2000 if QUICK else 100000
MIN_ERRORS = 20 if QUICK else 100
EB_N0_RANGE = np.arange(2.0, 7.5, 1.0 if QUICK else 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}


def sc_decoder(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None


print("SC 基线 (L=1)")
results_sc = run_simulation(
    N, K, EB_N0_RANGE, sc_decoder, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
)
all_results["SC (L=1)"] = results_sc
save_results_csv(results_sc, os.path.join(RESULTS_DIR, f"exp2_sc_N{N}_R0.5.csv"))

for L in L_LIST:
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

    def scl_decoder(llr_ch, _L=L):
        u_hat, pm = SCLDecoder(N, frozen_bits, list_size=_L, crc_length=0).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, os.path.join(RESULTS_DIR, f"exp2_scl_L{L}_N{N}_R0.5.csv"))

print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")


def cascl_decoder(llr_ch):
    u_hat, pm = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
    return u_hat, None


results_cascl = run_simulation(
    N,
    K,
    EB_N0_RANGE,
    cascl_decoder,
    "scl",
    MAX_FRAMES,
    MIN_ERRORS,
    crc_length=CRC_LENGTH,
    info_indices=info_idx,
)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
save_results_csv(results_cascl, os.path.join(RESULTS_DIR, f"exp2_cascl_L8_N{N}_R0.5.csv"))

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R={RATE})",
    os.path.join(RESULTS_DIR, "fig2_scl_bler.png"),
    shannon_limit_db=shannon_db,
)

labels = list(all_results.keys())
avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, avg_times)
ax.set_xlabel("Decoder")
ax.set_ylabel("Avg Decode Time (ms)")
ax.set_title(f"Decoding Time vs List Size (N={N})")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.pdf"))
plt.close()

print(f"\n实验二完成。结果保存至 {RESULTS_DIR}/")
