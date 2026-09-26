"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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
    rng = np.random.default_rng(seed)
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", max_frames))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", min_errors))
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

        if info_indices is None:
            info_idx = np.arange(K)
        else:
            info_idx = np.asarray(info_indices, dtype=int)

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            info_payload = rng.integers(0, 2, k_info)
            if crc_length > 0:
                payload = crc_encode(info_payload, crc_length)
                u[info_idx] = payload
            else:
                u[info_idx] = info_payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                from decoder_scl import crc_check

                block_err = not crc_check(u_hat[info_idx], crc_length)
                bit_err = np.sum(u_hat[info_idx[:k_info]] != info_payload)
            else:
                block_err = not np.array_equal(u_hat[info_idx], info_payload)
                bit_err = np.sum(u_hat[info_idx] != info_payload)

            num_frames += 1
            if block_err:
                num_errors += 1
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
