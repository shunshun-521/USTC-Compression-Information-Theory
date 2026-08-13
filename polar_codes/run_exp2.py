"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit
import matplotlib.pyplot as plt

# ========== 单元测试：L=1 SCL 等价 SC ==========
N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=int)
frozen_t[info_t] = 0
sigma_t = eb_n0_to_sigma(10.0, K_t / N_t)
rng = np.random.default_rng(1)
pm_err = 0
for _ in range(50):
    u = np.zeros(N_t, dtype=int)
    u[info_t] = rng.integers(0, 2, K_t)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma_t)
    u_sc = sc_decode(llr, frozen_t)
    u_scl, _ = SCLDecoder(N_t, frozen_t, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        pm_err += 1
assert pm_err == 0, f"路径度量校验失败: {pm_err}/50"

os.makedirs("results", exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_LIST = [2, 4, 8]
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}

def sc_decoder(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None

results_sc = run_simulation(
    N, K, EB_N0_RANGE, sc_decoder, "sc",
    MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
)
all_results["SC (L=1)"] = results_sc
save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

for L in L_LIST:
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")
    scl = SCLDecoder(N, frozen_bits, list_size=L, crc_length=0)

    def scl_decoder(llr_ch, _scl=scl):
        u_hat, pm = _scl.decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")
cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH)

def cascl_decoder(llr_ch):
    u_hat, pm = cascl.decode(llr_ch)
    return u_hat, None

results_cascl = run_simulation(
    N, K, EB_N0_RANGE, cascl_decoder, "scl",
    MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
    info_indices=info_idx, verbose=True,
)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results, f"SCL vs SC BLER (N={N}, R={RATE})",
    "results/fig2_scl_bler.png", shannon_limit_db=shannon_db,
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

print("\n实验二完成。")
