"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


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
    info_indices=None,
    verbose=True,
    seed=42,
):
    """蒙特卡洛仿真"""
    max_frames = _env_int("POLAR_MAX_FRAMES", max_frames if max_frames is not None else 100000)
    min_errors = _env_int("POLAR_MIN_ERRORS", min_errors if min_errors is not None else 100)

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        raise ValueError("info_indices must be provided")

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, K_info)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, K)

            u = np.zeros(N, dtype=int)
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

            if crc_length > 0:
                compare_idx = info_indices[:K_info]
            else:
                compare_idx = info_indices

            frame_err = not np.array_equal(u_hat[compare_idx], u[compare_idx])
            if frame_err:
                num_errors += 1
                num_bit_errors += np.count_nonzero(u_hat[compare_idx] != u[compare_idx])
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

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
