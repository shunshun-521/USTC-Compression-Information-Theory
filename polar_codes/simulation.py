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
    info_indices=None,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的 dict 列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        from construction import ga_construction

        info_indices, _, _ = ga_construction(N, K, 2.5)

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
            else:
                info_bits = rng.integers(0, 2, K)
                u[info_indices] = info_bits

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices[:k_info]]
                frame_err = not np.array_equal(decoded_info, info_bits)
                bit_err = np.count_nonzero(decoded_info != info_bits)
            else:
                frame_err = not np.array_equal(u_hat[info_indices], info_bits)
                bit_err = np.count_nonzero(u_hat[info_indices] != info_bits)

            num_errors += int(frame_err)
            num_bit_errors += bit_err
            num_frames += 1

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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results


def env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else default
