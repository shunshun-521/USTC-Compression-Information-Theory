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
    frozen_bits=None,
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
    K_info = K - crc_length

    if frozen_bits is None or info_indices is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            info_payload = rng.integers(0, 2, K_info)
            if crc_length > 0:
                payload_with_crc = crc_encode(info_payload, crc_length)
            else:
                payload_with_crc = info_payload

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload_with_crc

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            frame_error = not np.array_equal(u_hat[info_indices], payload_with_crc)
            if frame_error:
                num_errors += 1

            num_bit_errors += np.count_nonzero(
                u_hat[info_indices[:K_info]] != info_payload
            )
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
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
