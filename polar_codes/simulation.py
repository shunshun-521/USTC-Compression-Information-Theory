"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from utils import crc_encode_bits


def _fast_sim_defaults():
    if os.environ.get("POLAR_FAST_SIM", "").strip() in ("1", "true", "yes"):
        return 800, 10
    return None, None


def fast_eb_n0_range(default_range):
    if os.environ.get("POLAR_FAST_SIM", "").strip() in ("1", "true", "yes"):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return default_range


def fast_exp2_params(default_n, default_l_list):
    if os.environ.get("POLAR_FAST_SIM", "").strip() in ("1", "true", "yes"):
        return 256, [2, 4]
    return default_n, default_l_list


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
    """
    fast_max, fast_min = _fast_sim_defaults()
    if fast_max is not None:
        max_frames = min(max_frames, fast_max)
        min_errors = min(min_errors, fast_min)

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_indices = np.where(np.ones(N, dtype=int))[0][:K]
    info_indices = np.asarray(info_indices, dtype=int)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            info_payload = rng.integers(0, 2, size=K_info)
            if crc_length > 0:
                coded_info = crc_encode_bits(info_payload, crc_length)
            else:
                coded_info = info_payload
            u[info_indices] = coded_info

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            frame_error = not np.array_equal(u_hat[info_indices], coded_info)
            if frame_error:
                num_errors += 1
            num_bit_errors += np.count_nonzero(u_hat[info_indices] != coded_info)
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
