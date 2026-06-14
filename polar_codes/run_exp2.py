#!/usr/bin/env python3
"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)
QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

# ========== 单元测试 ==========
u_test = np.array([1, 0, 1, 1])
assert np.array_equal(polar_encode(u_test), [1, 1, 0, 1])

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
fb_t = np.ones(N_t, dtype=bool)
fb_t[info_t] = False
rng_t = np.random.default_rng(1)
sigma_t = eb_n0_to_sigma(4.0, 0.5)
for _ in range(20):
    uu = np.zeros(N_t, dtype=int)
    uu[info_t] = rng_t.integers(0, 2, size=K_t)
    llr_t = compute_llr(
        awgn_channel(bpsk_modulate(polar_encode(uu)), sigma_t, rng_t), sigma_t
    )
    u_sc = sc_decode(llr_t, fb_t)
    u_scl, _ = SCLDecoder(N_t, fb_t, list_size=1).decode(llr_t)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"
print("实验二：单元测试通过")

# ========== 参数 ==========
N = 128 if QUICK else 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_LIST = [2, 4] if QUICK else [2, 4, 8]
MAX_FRAMES = 2000 if QUICK else 100000
MIN_ERRORS = 20 if QUICK else 100
EB_N0_RANGE = np.arange(2.0, 5.5, 0.5) if QUICK else np.arange(1.0, 5.5, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
fb_bool = frozen_bits.astype(bool)

all_results = {}


def sc_decoder(llr_ch):
    return sc_decode(llr_ch, fb_bool), None


print("\nSC 基线 (L=1)")
results_sc = run_simulation(
    N, K, EB_N0_RANGE, sc_decoder, "sc", MAX_FRAMES, MIN_ERRORS, verbose=True
)
all_results["SC (L=1)"] = results_sc
save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

for L in L_LIST:
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

    def scl_decoder(llr_ch, _L=L):
        u_hat, _ = SCLDecoder(N, fb_bool, list_size=_L, crc_length=0).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl", MAX_FRAMES, MIN_ERRORS, verbose=True
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")


def cascl_decoder(llr_ch):
    u_hat, _ = SCLDecoder(N, fb_bool, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
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
    verbose=True,
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

print("\n实验二完成。")
