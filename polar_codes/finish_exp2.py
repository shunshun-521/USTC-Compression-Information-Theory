"""完成实验二剩余部分（L=8、CA-SCL、绘图）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

N = 512
K = N // 2
CRC_LENGTH = 8
EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)
MAX_FRAMES = 100
MIN_ERRORS = 10

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

for L, path in [(8, "results/exp2_scl_L8_N512_R0.5.csv")]:
    if not os.path.exists(path):
        print(f"SCL L={L} 快速仿真...")

        def scl_decoder(llr_ch, _L=L):
            u, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
            return u, None

        results = run_simulation(
            N, K, EB_N0_RANGE, scl_decoder, "scl",
            MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        save_results_csv(results, path)
    all_results[f"SCL (L={L})"] = load_results_csv(path)

cascl_path = "results/exp2_cascl_L8_N512_R0.5.csv"
if not os.path.exists(cascl_path):
    print("CA-SCL L=8 快速仿真...")

    def cascl_decoder(llr_ch):
        u, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
        return u, None

    results = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, "scl",
        MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, cascl_path)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = load_results_csv(cascl_path)

save_results_csv(all_results["SCL (L=4)"], "results/exp2_scl_N512_R0.5.csv")

shannon_db = find_capacity_limit(0.5)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R=0.5)",
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
