"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode


def _sim_limits():
    fast = os.environ.get("POLAR_FAST_SIM", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not fast else "2000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not fast else "20"))
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
    verbose=True,
    seed=42,
    info_indices=None,
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

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                if info_indices is None:
                    raise ValueError("info_indices required when crc_length > 0")
                u[info_indices] = payload
                check_bits = info_bits
            else:
                u = np.zeros(N, dtype=int)
                if info_indices is None:
                    info_indices_local = np.where(np.arange(N) >= N - K)[0]
                else:
                    info_indices_local = info_indices
                u[info_indices_local] = rng.integers(0, 2, K)
                check_bits = u[info_indices_local]

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
                if decoder_type == "bp":
                    total_iters += aux if aux is not None else 0
            else:
                u_hat = decode_out

            if info_indices is None:
                info_indices_local = np.where(np.arange(N) >= N - K)[0]
            else:
                info_indices_local = info_indices

            if crc_length > 0:
                decoded_info = u_hat[info_indices_local][:k_info]
                frame_error = not np.array_equal(decoded_info, check_bits)
                bit_errors = np.sum(decoded_info != check_bits)
            else:
                decoded_info = u_hat[info_indices_local]
                frame_error = not np.array_equal(decoded_info, check_bits)
                bit_errors = np.sum(decoded_info != check_bits)

            num_frames += 1
            num_errors += int(frame_error)
            num_bit_errors += int(bit_errors)

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
