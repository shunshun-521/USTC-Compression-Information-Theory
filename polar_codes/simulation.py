"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode
from encoder import polar_encode


def _env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else default


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
):
    """蒙特卡洛仿真。"""
    if max_frames is None:
        max_frames = _env_int("POLAR_MAX_FRAMES", 100000)
    if min_errors is None:
        min_errors = _env_int("POLAR_MIN_ERRORS", 100)
    if os.environ.get("POLAR_QUICK") == "1":
        max_frames = min(max_frames, 5000)
        min_errors = min(min_errors, 20)

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        from construction import ga_construction
        info_indices, _, _ = ga_construction(N, K, 2.5)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=np.int8)

            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=K_info, dtype=np.int8)
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
            else:
                u[info_indices] = rng.integers(0, 2, size=K, dtype=np.int8)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                frame_err = not np.array_equal(u_hat[info_indices][:K_info], info_bits)
                bit_err = np.sum(u_hat[info_indices][:K_info] != info_bits)
            else:
                frame_err = not np.array_equal(u_hat[info_indices], u[info_indices])
                bit_err = np.sum(u_hat[info_indices] != u[info_indices])

            num_errors += int(frame_err)
            num_bit_errors += int(bit_err)
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
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
