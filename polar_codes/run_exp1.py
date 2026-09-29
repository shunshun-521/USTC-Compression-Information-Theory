"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    load_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

os.makedirs("results", exist_ok=True)

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
if os.environ.get("POLAR_N_LIST"):
    N_LIST = [int(x.strip()) for x in os.environ["POLAR_N_LIST"].split(",") if x.strip()]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(
    float(os.environ.get("POLAR_EB_MIN", "0.0")),
    float(os.environ.get("POLAR_EB_MAX", "5.5")),
    float(os.environ.get("POLAR_EB_STEP", "0.25")),
)

if os.environ.get("POLAR_SKIP_N1024", "0") == "1":
    N_LIST = [n for n in N_LIST if n != 1024]

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt", rate=RATE)

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits,
        verbose=True,
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

# 绘图时合并 results/ 下已有的 exp1 CSV（便于分次补跑 N=1024 等）
plot_results = {}
for n in [256, 512, 1024]:
    csv_path = f"results/exp1_sc_N{n}_R0.5.csv"
    if os.path.isfile(csv_path):
        plot_results[f"SC, N={n}, K={n // 2}"] = load_results_csv(csv_path)
plot_bler_curves(
    plot_results or all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
