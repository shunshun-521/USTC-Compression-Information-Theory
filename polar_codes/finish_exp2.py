"""补全实验二：CA-SCL 与绘图。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import load_results_csv, plot_bler_curves, save_results_csv, find_capacity_limit

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)
MAX_FRAMES = 100000
MIN_ERRORS = 100

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}
for label, path in [
    ("SC (L=1)", "results/exp2_sc_N512_R0.5.csv"),
    ("SCL (L=2)", "results/exp2_scl_L2_N512_R0.5.csv"),
    ("SCL (L=4)", "results/exp2_scl_L4_N512_R0.5.csv"),
    ("SCL (L=8)", "results/exp2_scl_L8_N512_R0.5.csv"),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)


def cascl_decoder(llr_ch):
    u_hat, pm = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(
        llr_ch
    )
    return u_hat, None


print("CA-SCL 仿真...")
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
    design_eb_n0_db=DESIGN_EBN0,
)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R={RATE})",
    "results/fig2_scl_bler.png",
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
plt.savefig("results/fig2_decode_time.png", dpi=150)
plt.savefig("results/fig2_decode_time.pdf")
plt.close()
print("实验二补全完成。")
