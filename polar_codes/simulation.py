"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode
from decoder_scl import crc_encode


def _sim_params(max_frames, min_errors):
    """支持快速仿真环境变量"""
    if os.environ.get("POLAR_FAST_SIM") == "1" or os.environ.get("POLAR_QUICK") == "1":
        return min(max_frames, 2000), min(min_errors, 10)
    max_override = os.environ.get("POLAR_MAX_FRAMES")
    min_override = os.environ.get("POLAR_MIN_ERRORS")
    if max_override:
        max_frames = int(max_override)
    if min_override:
        min_errors = int(min_override)
    return max_frames, min_errors


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
    design_eb_n0_db=2.5,
    info_indices=None,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    max_frames, min_errors = _sim_params(max_frames, min_errors)
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
    info_indices = np.asarray(info_indices, dtype=int)

    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_indices] = 0

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=np.int8)

            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=k_info, dtype=np.int8)
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
            else:
                info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
                u[info_indices] = info_bits

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices]
                block_err = not crc_check_bits(decoded_info, crc_length)
                bit_err = int(np.sum(info_bits != decoded_info[:k_info]))
            else:
                block_err = not np.array_equal(u_hat[info_indices], info_bits)
                bit_err = int(np.sum(info_bits != u_hat[info_indices]))

            num_errors += int(block_err)
            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results


def crc_check_bits(bits, crc_length):
    from decoder_scl import crc_check

    return crc_check(bits, crc_length)
