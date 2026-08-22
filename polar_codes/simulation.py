"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import align_llr_for_decoder, polar_encode


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    info_indices=None,
    verbose=True,
    seed=42,
):
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices = np.arange(N)
    info_indices = np.asarray(info_indices)
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
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, size=k_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices[:K]] = payload
            else:
                info_bits = rng.integers(0, 2, size=K)
                u = np.zeros(N, dtype=int)
                u[info_indices[:K]] = info_bits

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = align_llr_for_decoder(compute_llr(y, sigma))

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                from decoder_scl import crc_check

                decoded_info = u_hat[info_indices[:K]]
                frame_ok = crc_check(decoded_info, crc_length)
                if not frame_ok:
                    num_errors += 1
                    num_bit_errors += np.sum(info_bits != decoded_info[:k_info])
            else:
                sent_info = u[info_indices[:K]]
                got_info = u_hat[info_indices[:K]]
                if not np.array_equal(sent_info, got_info):
                    num_errors += 1
                    num_bit_errors += np.sum(sent_info != got_info)

            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info > 0 else 1.0
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


def get_sim_params():
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not quick else "2000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not quick else "20"))
    return max_frames, min_errors
