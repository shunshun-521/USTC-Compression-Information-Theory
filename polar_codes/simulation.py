"""
蒙特卡洛仿真主循环
"""
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
    info_indices=None,
    frozen_bits=None,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=bool)
        if info_indices is not None:
            frozen_bits[info_indices] = False

    info_mask = ~np.asarray(frozen_bits, dtype=bool)
    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            info_bits = rng.integers(0, 2, K_info)
            u_sent = np.zeros(N, dtype=int)

            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u_sent[info_mask] = payload
            else:
                u_sent[info_mask] = info_bits

            x = polar_encode(u_sent)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                frame_error = not np.array_equal(u_hat[info_mask], u_sent[info_mask])
                bit_err = np.sum(u_hat[info_mask][:K_info] != info_bits)
            else:
                frame_error = not np.array_equal(u_hat[info_mask], u_sent[info_mask])
                bit_err = np.sum(u_hat[info_mask] != u_sent[info_mask])

            num_errors += int(frame_error)
            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info)
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
