"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from construction import ga_construction


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
    design_eb_n0_db=2.5,
):
    """
    蒙特卡洛仿真。
    """
    if os.environ.get("POLAR_QUICK") == "1":
        max_frames = min(max_frames, 2000)
        min_errors = min(min_errors, 20)

    rng = np.random.default_rng(seed)
    rate = K / N
    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
    info_indices = np.asarray(info_indices, dtype=int)

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
                payload = rng.integers(0, 2, k_info, dtype=np.int8)
                from decoder_scl import crc_encode

                msg = crc_encode(payload, crc_length)
            else:
                msg = rng.integers(0, 2, K, dtype=np.int8)

            u = np.zeros(N, dtype=np.int8)
            u[info_indices] = msg

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            u_hat = np.asarray(u_hat, dtype=np.int8)
            if crc_length > 0:
                frame_err = not crc_check_payload(u_hat, info_indices, crc_length, payload)
            else:
                frame_err = np.any(u_hat[info_indices] != msg)

            if frame_err:
                num_errors += 1
            num_bit_errors += np.sum(u_hat[info_indices][:k_info] != msg[:k_info])
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info else 0.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time*1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results


def crc_check_payload(u_hat, info_indices, crc_length, payload):
    from decoder_scl import crc_check

    bits = u_hat[info_indices]
    if len(bits) < len(payload) + crc_length:
        return False
    if not np.array_equal(bits[: len(payload)], payload):
        return False
    return crc_check(bits[: len(payload) + crc_length], crc_length)
