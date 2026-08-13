"""快速生成示例仿真结果（用于验证流程）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs("results", exist_ok=True)
EB = np.array([1.5, 2.5, 3.5])
RATE = 0.5
DESIGN = 2.5
MAX_F = int(os.environ.get("POLAR_MAX_FRAMES", 200))
MIN_E = int(os.environ.get("POLAR_MIN_ERRORS", 5))

save_frozen_set_info([256, 512, 1024], None, DESIGN, "results/frozen_sets.txt")

all1 = {}
for N in [256, 512, 1024]:
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fz = np.ones(N, dtype=int)
    fz[info] = 0
    r = run_simulation(
        N, K, EB, lambda l, f=fz: (sc_decode(l, f), None), "sc",
        MAX_F, MIN_E, info_indices=info, verbose=True,
    )
    save_results_csv(r, f"results/exp1_sc_N{N}_R0.5.csv")
    all1[f"N={N}"] = r
plot_bler_curves(all1, "SC", "results/fig1_sc_bler.png", find_capacity_limit(RATE))

N = 512
K = 256
info, _, _ = ga_construction(N, K, DESIGN)
fz = np.ones(N, dtype=int)
fz[info] = 0
all2 = {}
all2["SC"] = run_simulation(
    N, K, EB, lambda l: (sc_decode(l, fz), None), "sc",
    MAX_F, MIN_E, info_indices=info, verbose=True,
)
save_results_csv(all2["SC"], "results/exp2_sc_N512_R0.5.csv")
for L in [2, 4, 8]:
    r = run_simulation(
        N, K, EB,
        lambda l, _L=L: (SCLDecoder(N, fz, _L, info_indices=info).decode(l)[0], None),
        "scl", MAX_F, MIN_E, info_indices=info, verbose=True,
    )
    all2[f"SCL L={L}"] = r
    save_results_csv(r, f"results/exp2_scl_L{L}_N512_R0.5.csv")
all2["CA-SCL"] = run_simulation(
    N, K, EB,
    lambda l: (SCLDecoder(N, fz, 8, 8, info).decode(l)[0], None),
    "scl", MAX_F, MIN_E, crc_length=8, info_indices=info, verbose=True,
)
save_results_csv(all2["CA-SCL"], "results/exp2_cascl_L8_N512_R0.5.csv")
plot_bler_curves(all2, "SCL", "results/fig2_scl_bler.png", find_capacity_limit(RATE))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    list(all2.keys()),
    [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all2.values()],
)
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("results/fig2_decode_time.png", dpi=150)
plt.savefig("results/fig2_decode_time.pdf")
plt.close()

for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fz = np.ones(N, dtype=int)
    fz[info] = 0
    bp = BPDecoder(N, fz, 50)
    all3 = {
        "SC": run_simulation(
            N, K, EB, lambda l: (sc_decode(l, fz), None), "sc",
            MAX_F, MIN_E, info_indices=info, verbose=True,
        ),
        "SCL": run_simulation(
            N, K, EB,
            lambda l: (SCLDecoder(N, fz, 4, info_indices=info).decode(l)[0], None),
            "scl", MAX_F, MIN_E, info_indices=info, verbose=True,
        ),
    }

    def bp_dec(llr, _bp=bp):
        u, iters = _bp.decode(llr)
        return u, iters

    all3["BP"] = run_simulation(
        N, K, EB, bp_dec, "bp", MAX_F, MIN_E, info_indices=info, verbose=True,
    )
    save_results_csv(all3["SC"], f"results/exp3_sc_N{N}_R0.5.csv")
    save_results_csv(all3["SCL"], f"results/exp3_scl_N{N}_R0.5.csv")
    save_results_csv(all3["BP"], f"results/exp3_bp_N{N}_R0.5.csv")
    plot_bler_curves(
        all3, f"N={N}", f"results/fig3_bp_N{N}_bler.png", find_capacity_limit(RATE),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [x["eb_n0_db"] for x in all3["BP"]],
        [x["avg_iters"] for x in all3["BP"]],
        "o-",
        color="purple",
    )
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

print("generate_sample_results.py 完成")
