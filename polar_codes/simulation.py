"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_scl import crc_encode
from encoder import polar_encode


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        info_indices = info_idx

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=K_info, dtype=np.int8)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, size=K, dtype=np.int8)

            u = np.zeros(N, dtype=np.int8)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            decoded_info = u_hat[info_indices]
            if crc_length > 0:
                frame_error = not np.array_equal(decoded_info[:K_info], info_bits)
                bit_errors = np.sum(decoded_info[:K_info] != info_bits)
            else:
                frame_error = not np.array_equal(decoded_info, payload)
                bit_errors = np.sum(decoded_info != payload)

            if frame_error:
                num_errors += 1
            num_bit_errors += bit_errors
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info)
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

        result = {
            "eb_n0_db": eb_n0_db,
            "bler": bler,
            "ber": ber,
            "num_errors": num_errors,
            "num_frames": num_frames,
            "avg_decode_time": avg_time,
            "avg_iters": avg_iters,
        }
        results.append(result)

        if verbose:
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters else "")
            )

    return results


def get_sim_params():
    """从环境变量读取仿真参数（用于快速测试）"""
    fast = os.environ.get("POLAR_FAST_SIM", "0") == "1"
    max_frames = int(
        os.environ.get("POLAR_MAX_FRAMES", "100000" if not fast else "3000")
    )
    min_errors = int(
        os.environ.get("POLAR_MIN_ERRORS", "100" if not fast else "20")
    )
    return max_frames, min_errors, fast


def get_eb_n0_range(full_range, fast=False):
    """快速模式下使用更少的信噪比采样点"""
    if fast:
        return np.arange(full_range[0], full_range[-1] + 0.01, 0.5)
    return full_range
