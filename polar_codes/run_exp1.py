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
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    from channel import bpsk_modulate, compute_llr
    from decoder_scl import SCLDecoder

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    errors = 0
    for _ in range(100):
        ut = np.zeros(N, dtype=int)
        ut[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(ut)), 1e-8)
        if not np.array_equal(sc_decode(llr, frozen_bits, use_exact=True), ut):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100"

    mism = 0
    for _ in range(50):
        ut = np.zeros(N, dtype=int)
        ut[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(ut)), 1e-8)
        if not np.array_equal(
            sc_decode(llr, frozen_bits, use_exact=True),
            SCLDecoder(N, frozen_bits, list_size=1, use_exact=True).decode(llr)[0],
        ):
            mism += 1
    assert mism == 0, f"SCL(L=1) 与 SC 不一致: {mism}/50"
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

if __name__ == "__main__":
    run_unit_tests()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")

        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        fb = frozen_bits.astype(bool)

        def decoder(llr_ch):
            return sc_decode(llr_ch, fb, use_exact=True), None

        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            design_eb_n0_db=DESIGN_EBN0,
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
