"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
- 仿真并绘制 BLER-Eb/N0 曲线
- 添加 BPSK 信道容量限
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode, verify_sc_decoders
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

assert verify_sc_decoders(N=64, num_frames=100, eb_n0_db=10.0), "SC 译码校验失败"
print("单元测试通过。")

os.makedirs('results', exist_ok=True)

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 20000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(4.0, 10.5, 0.5)

# ========== 保存信息位/冻结位集合 ==========
save_frozen_set_info(N_LIST, None, DESIGN_EBN0, 'results/frozen_sets.txt')

# ========== 仿真循环 ==========
all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, llr_means = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, _frozen=frozen_bits.copy()):
        return sc_decode(llr_ch, _frozen), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type='sc',
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        verbose=True,
    )

    label = f'SC, N={N}, K={K}'
    all_results[label] = results
    save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

# ========== 绘图 ==========
shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f'SC Decoder BLER vs Eb/N0 (R={RATE})',
    save_path='results/fig1_sc_bler.png',
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
