"""
蒙特卡洛仿真主循环
"""
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
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None or frozen_bits is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        if frozen_bits is None:
            frozen_bits = np.ones(N, dtype=int)
            frozen_bits[info_idx] = 0
        if info_indices is None:
            info_indices = info_idx

    info_indices = np.asarray(info_indices)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    K_info = K - crc_length

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

                info_payload = rng.integers(0, 2, size=K_info)
                payload_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload_with_crc
            else:
                info_payload = rng.integers(0, 2, size=K)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices]
                from decoder_scl import crc_check

                frame_error = not crc_check(decoded_info, crc_length)
                bit_err = np.sum(decoded_info[:K_info] != info_payload)
            else:
                decoded_info = u_hat[info_indices]
                frame_error = not np.array_equal(decoded_info, info_payload)
                bit_err = np.sum(decoded_info != info_payload)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_err

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
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
