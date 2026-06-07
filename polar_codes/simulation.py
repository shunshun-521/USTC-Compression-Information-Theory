"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode
from encoder import polar_encode


def _sim_limits():
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not quick else "2000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not quick else "20"))
    return max_frames, min_errors


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    info_indices=None,
    verbose=True,
    seed=42,
):
    """蒙特卡洛仿真"""
    default_max, default_min = _sim_limits()
    if max_frames is None:
        max_frames = default_max
    if min_errors is None:
        min_errors = default_min

    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length
    results = []

    if info_indices is None:
        info_indices = np.arange(N - K, N)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                msg = rng.integers(0, 2, k_info)
                payload = crc_encode(msg, crc_length)
            else:
                payload = rng.integers(0, 2, K)

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            frame_error = not np.array_equal(u_hat[info_indices], payload)
            if frame_error:
                num_errors += 1
            num_bit_errors += np.count_nonzero(u_hat[info_indices] != payload)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

        result = {
            "eb_n0_db": float(eb_n0_db),
            "bler": float(bler),
            "ber": float(ber),
            "num_errors": int(num_errors),
            "num_frames": int(num_frames),
            "avg_decode_time": float(avg_time),
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
