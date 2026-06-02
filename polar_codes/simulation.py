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
    design_ebn0_db=2.5,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    info_idx, _, _ = ga_construction(N, K, design_ebn0_db, rate)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    K_info = K - crc_length
    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                from decoder_scl import crc_encode

                payload = rng.integers(0, 2, K_info)
                coded = crc_encode(payload, crc_length)
                u[info_idx[: K_info + crc_length]] = coded
            else:
                u[info_idx] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                err_bits = np.sum(u_hat[info_idx[:K_info]] != u[info_idx[:K_info]])
            else:
                err_bits = np.sum(u_hat[info_idx] != u[info_idx])

            num_bit_errors += err_bits
            if err_bits > 0:
                num_errors += 1
            num_frames += 1

        bler = num_errors / max(num_frames, 1)
        ber = num_bit_errors / max(num_frames * K_info, 1)
        avg_time = total_decode_time / max(num_frames, 1)
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
                f"| AvgTime={avg_time*1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
