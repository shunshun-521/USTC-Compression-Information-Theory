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
from encoder import polar_encode
from channel import awgn_channel, bpsk_modulate, compute_llr
from decoder_scl import crc_encode, crc_check
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

# ========== 单元测试（与实验一相同） ==========
u = np.array([1, 0, 1, 1])
assert np.array_equal(polar_encode(u), [1, 1, 0, 1])

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=bool)
frozen_t[info_t] = False
rng = np.random.default_rng(0)
for _ in range(100):
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = rng.integers(0, 2, size=K_t)
    llr_t = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), 0.001, rng), 0.001)
    assert np.array_equal(sc_decode(llr_t, frozen_t), u_t)

for _ in range(20):
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = rng.integers(0, 2, size=K_t)
    llr_t = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), 0.001, rng), 0.001)
    u_scl, _ = SCLDecoder(N_t, frozen_t, list_size=1).decode(llr_t)
    assert np.array_equal(sc_decode(llr_t, frozen_t), u_scl)

bits = np.array([1, 0, 1, 1, 0, 0, 1])
assert crc_check(crc_encode(bits, 8), 8)
print("单元测试通过。\n")

os.makedirs("results", exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_LIST = [int(x) for x in os.environ.get("POLAR_L_LIST", "2,4,8").split(",")]
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_MIN = float(os.environ.get("POLAR_EB_MIN", "1.0"))
EB_N0_MAX = float(os.environ.get("POLAR_EB_MAX", "5.25"))
EB_N0_STEP = float(os.environ.get("POLAR_EB_STEP", "0.25"))
EB_N0_RANGE = np.arange(EB_N0_MIN, EB_N0_MAX, EB_N0_STEP)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=bool)
frozen_bits[info_idx] = False

all_results = {}


def sc_decoder(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None


print("SC 基线 (L=1)")
results_sc = run_simulation(
    N, K, EB_N0_RANGE, sc_decoder, "sc", MAX_FRAMES, MIN_ERRORS,
    info_indices=info_idx, verbose=True,
)
all_results["SC (L=1)"] = results_sc
save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

for L in L_LIST:
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

    def scl_decoder(llr_ch, _L=L):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=_L, crc_length=0).decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")


def cascl_decoder(llr_ch):
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
    return u_hat, None


results_cascl = run_simulation(
    N, K, EB_N0_RANGE, cascl_decoder, "scl", MAX_FRAMES, MIN_ERRORS,
    crc_length=CRC_LENGTH, info_indices=info_idx, verbose=True,
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
