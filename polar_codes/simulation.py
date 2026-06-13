"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode
from decoder_scl import crc_encode


def _env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else default


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
    design_eb_n0_db=2.5,
):
    """
    蒙特卡洛仿真。
    """
    if max_frames is None:
        max_frames = _env_int("POLAR_MAX_FRAMES", 100000)
    if min_errors is None:
        min_errors = _env_int("POLAR_MIN_ERRORS", 100)

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None or frozen_bits is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0

    info_positions = np.where(frozen_bits == 0)[0]
    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            info_bits = rng.integers(0, 2, size=K_info)

            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u[info_positions[: len(payload)]] = payload
            else:
                u[info_positions] = info_bits

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
                decoded_info = u_hat[info_positions[:K_info]]
                frame_error = not np.array_equal(decoded_info, info_bits)
                bit_err = np.sum(decoded_info != info_bits)
            else:
                decoded_info = u_hat[info_positions]
                frame_error = not np.array_equal(decoded_info, info_bits)
                bit_err = np.sum(decoded_info != info_bits)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_err

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
