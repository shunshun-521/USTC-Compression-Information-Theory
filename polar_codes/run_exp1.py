"""
实验一：SC 译码基础仿真
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """模块正确性校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u_t = np.zeros(N, dtype=int)
        u_t[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u_t)) + rng.normal(0, sigma, N), sigma
        )
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_t[info_idx]):
            errors += 1
    assert errors <= 2, f"SC 高信噪比校验失败: {errors}/100 帧错误"

    from decoder_scl import SCLDecoder

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    rng = np.random.default_rng(0)
    u_t = np.zeros(N, dtype=int)
    u_t[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(
        bpsk_modulate(polar_encode(u_t)) + rng.normal(0, sigma, N), sigma
    )
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, sc_decode(llr, frozen_bits)), "L=1 SCL 应等价于 SC"
    print("单元测试通过。")


FAST = os.environ.get("POLAR_FAST", "0") == "1"
MAX_FRAMES = 2000 if FAST else 100000
MIN_ERRORS = 20 if FAST else 100
EB_N0_RANGE = (
    np.arange(0.0, 8.5, 1.0) if FAST else np.arange(0.0, 5.5, 0.25)
)
N_LIST_FAST = [256, 512]

os.makedirs("results", exist_ok=True)
run_unit_tests()

N_LIST = N_LIST_FAST if FAST else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'='*60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'='*60}")

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    def decoder(llr_ch, _fb=frozen_bits):
        return sc_decode(llr_ch, _fb), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits.astype(int),
        verbose=True,
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。结果保存至 results/ 目录。")
