"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


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

    返回每个 Eb/N0 点的结果字典列表。
    """
    if os.environ.get("POLAR_FAST_SIM") == "1":
        max_frames = min(max_frames, 2000)
        min_errors = min(min_errors, 20)

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    K_info = K - crc_length

    if info_indices is None:
        raise ValueError("info_indices must be provided")

    info_indices = np.asarray(info_indices)

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
                msg = rng.integers(0, 2, K_info)
                payload = crc_encode(msg, crc_length)
                u[info_indices] = payload
            else:
                u[info_indices] = rng.integers(0, 2, len(info_indices))

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
                if decoder_type == "bp" and aux is not None:
                    total_iters += aux
            else:
                u_hat = decode_out

            bit_err = np.sum(u[info_indices[:K_info]] != u_hat[info_indices[:K_info]])
            frame_err = bit_err > 0
            num_bit_errors += bit_err
            num_errors += int(frame_err)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
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
