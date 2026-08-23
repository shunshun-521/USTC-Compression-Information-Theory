"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from construction import ga_construction


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type='sc',
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
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, 2.5)
    info_indices = np.asarray(info_indices, dtype=int)
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
                info_bits = rng.integers(0, 2, size=k_info, dtype=np.int8)
                from decoder_scl import crc_encode

                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = payload
            else:
                u = np.zeros(N, dtype=np.int8)
                u[info_indices] = rng.integers(0, 2, size=K, dtype=np.int8)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == 'bp' and aux is not None:
                total_iters += aux

            if crc_length > 0:
                payload_hat = u_hat[info_indices]
                from decoder_scl import crc_check

                frame_ok = crc_check(payload_hat, crc_length)
                bit_err = np.sum(payload_hat[:k_info] != info_bits)
            else:
                frame_ok = np.array_equal(u_hat[info_indices], u[info_indices])
                bit_err = np.sum(u_hat[info_indices] != u[info_indices])

            num_frames += 1
            if not frame_ok:
                num_errors += 1
            num_bit_errors += bit_err

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' and num_frames else None

        result = {
            'eb_n0_db': float(eb_n0_db),
            'bler': float(bler),
            'ber': float(ber),
            'num_errors': int(num_errors),
            'num_frames': int(num_frames),
            'avg_decode_time': float(avg_time),
            'avg_iters': avg_iters,
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
