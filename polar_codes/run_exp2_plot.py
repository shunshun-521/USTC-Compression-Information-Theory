"""根据已有 CSV 重新绘制实验二图表。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_results_csv, plot_bler_curves, find_capacity_limit
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

N = 512
RATE = 0.5
all_results = {}

for path, label in [
    ("results/exp2_sc_N512_R0.5.csv", "SC (L=1)"),
    ("results/exp2_scl_L2_N512_R0.5.csv", "SCL (L=2)"),
    ("results/exp2_scl_L4_N512_R0.5.csv", "SCL (L=4)"),
    ("results/exp2_scl_L8_N512_R0.5.csv", "SCL (L=8)"),
    ("results/exp2_cascl_L8_N512_R0.5.csv", "CA-SCL (L=8, CRC=8)"),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

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
print("实验二图表已更新。")
