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
    info_indices=None,
    verbose=True,
    seed=42,
):
    """蒙特卡洛仿真。"""
    if max_frames is None:
        max_frames = _env_int("POLAR_MAX_FRAMES", 100000)
    if min_errors is None:
        min_errors = _env_int("POLAR_MIN_ERRORS", 100)

    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length
    results = []

    if info_indices is None:
        raise ValueError("info_indices is required for simulation")

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            info_bits = rng.integers(0, 2, size=k_info)
            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = info_bits

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None:
                total_iters += aux

            sent_info = u[info_indices]
            decoded_info = u_hat[info_indices]
            frame_error = not np.array_equal(sent_info, decoded_info)
            num_bit_errors += np.count_nonzero(sent_info != decoded_info)
            num_errors += int(frame_error)
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames else 1.0
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
