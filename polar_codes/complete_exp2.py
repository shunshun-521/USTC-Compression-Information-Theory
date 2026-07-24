"""快速完成实验二剩余部分（L=8 与 CA-SCL）"""
import os
import matplotlib.pyplot as plt
import numpy as np

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

N = 512
K = N // 2
RATE = 0.5
EB_N0_RANGE = np.array([3.0, 4.0, 5.0])
MAX_FRAMES = 30
MIN_ERRORS = 5

info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}
for label, path in [
    ("SC (L=1)", "results/exp2_sc_N512_R0.5.csv"),
    ("SCL (L=2)", "results/exp2_scl_L2_N512_R0.5.csv"),
    ("SCL (L=4)", "results/exp2_scl_L4_N512_R0.5.csv"),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

for L in [8]:
    def dec(llr, _L=L):
        u, _ = SCLDecoder(N, frozen_bits, _L).decode(llr)
        return u, None

    results = run_simulation(
        N, K, EB_N0_RANGE, dec, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

def cascl_dec(llr):
    u, _ = SCLDecoder(N, frozen_bits, 8, crc_length=8).decode(llr)
    return u, None

results_cascl = run_simulation(
    N, K, EB_N0_RANGE, cascl_dec, "scl", MAX_FRAMES, MIN_ERRORS,
    crc_length=8, info_indices=info_idx,
)
all_results["CA-SCL (L=8, CRC=8)"] = results_cascl
save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R={RATE})",
    "results/fig2_scl_bler.png",
    shannon_db,
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
print("实验二补全完成")
