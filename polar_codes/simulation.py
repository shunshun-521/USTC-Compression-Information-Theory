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
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    """
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", max_frames or 100000))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", min_errors or 100))

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None or frozen_bits is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        info_indices = info_idx
    else:
        info_indices = np.asarray(info_indices)
        frozen_bits = np.asarray(frozen_bits)

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=k_info, dtype=np.int8)
                payload_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = payload_with_crc
            else:
                info_payload = rng.integers(0, 2, size=K, dtype=np.int8)
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = info_payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_payload = u_hat[info_indices]
                frame_error = not np.array_equal(decoded_payload[:k_info], info_payload)
                bit_errors = np.count_nonzero(decoded_payload[:k_info] != info_payload)
            else:
                decoded_info = u_hat[info_indices]
                frame_error = not np.array_equal(decoded_info, info_payload)
                bit_errors = np.count_nonzero(decoded_info != info_payload)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_errors

        bler = num_errors / num_frames if num_frames else 0.0
        ber_den = num_frames * k_info if k_info > 0 else num_frames * K
        ber = num_bit_errors / ber_den if ber_den else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

        result = {
            "eb_n0_db": float(eb_n0_db),
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
