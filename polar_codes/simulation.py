"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


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
    if os.environ.get("POLAR_QUICK") == "1":
        max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "5000"))
        min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "20"))

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, K)

            u = np.zeros(N, dtype=int)
            if info_indices is None:
                raise ValueError("info_indices is required")
            u[info_indices] = payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                frame_err = not crc_check_compat(u_hat, info_indices, info_bits, crc_length)
                bit_err = np.sum(u_hat[info_indices][:k_info] != info_bits)
            else:
                frame_err = np.any(u_hat[info_indices] != payload)
                bit_err = np.sum(u_hat[info_indices] != payload)

            num_frames += 1
            num_errors += int(frame_err)
            num_bit_errors += bit_err

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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results


def crc_check_compat(u_hat, info_indices, info_bits, crc_length):
    from decoder_scl import crc_check

    received = u_hat[info_indices]
    return crc_check(received, crc_length) and np.array_equal(
        received[: len(info_bits)], info_bits
    )
