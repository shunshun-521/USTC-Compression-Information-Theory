"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode
from encoder import polar_encode


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
    results = []

    if frozen_bits is None or info_indices is None:
        from construction import ga_construction

        info_indices, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=np.int32)
        frozen_bits[info_indices] = 0

    info_indices = np.asarray(info_indices, dtype=np.int64)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int32)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=K_info, dtype=np.int8)
                coded_info = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = coded_info
            else:
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = rng.integers(0, 2, size=K, dtype=np.int8)

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
                payload_hat = u_hat[info_indices][:K_info]
                frame_error = not np.array_equal(payload_hat, info_payload)
                bit_err = np.count_nonzero(payload_hat != info_payload)
            else:
                payload_hat = u_hat[info_indices]
                payload_sent = u[info_indices]
                frame_error = not np.array_equal(payload_hat, payload_sent)
                bit_err = np.count_nonzero(payload_hat != payload_sent)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_err

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
        avg_time = total_decode_time / num_frames
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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
