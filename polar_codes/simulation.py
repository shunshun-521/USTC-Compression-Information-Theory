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
    N, K, eb_n0_db_list, decoder,
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
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    k_info = K - crc_length

    if info_indices is None:
        raise ValueError("info_indices must be provided")

    info_indices = np.asarray(info_indices, dtype=int)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=k_info, dtype=int)
                info_with_crc = crc_encode(info_payload, crc_length)
            else:
                info_with_crc = rng.integers(0, 2, size=K, dtype=int)
                info_payload = info_with_crc

            u = np.zeros(N, dtype=int)
            u[info_indices] = info_with_crc

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
                decoded_info = u_hat[info_indices][:k_info]
                frame_error = not np.array_equal(decoded_info, info_payload)
                bit_errors = np.sum(decoded_info != info_payload)
            else:
                decoded_info = u_hat[info_indices]
                frame_error = not np.array_equal(decoded_info, info_with_crc)
                bit_errors = np.sum(decoded_info != info_with_crc)

            num_errors += int(frame_error)
            num_bit_errors += bit_errors
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


def fast_sim_env():
    """是否启用快速仿真模式（自动化/CI 用）"""
    return os.environ.get("POLAR_FAST_SIM", "0") == "1"
