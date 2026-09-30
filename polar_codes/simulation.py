"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


def _sim_limits():
    if os.environ.get("POLAR_FAST", "").strip() in ("1", "true", "yes"):
        return 5000, 20
    return 100000, 100


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
):
    default_max, default_min = _sim_limits()
    if max_frames is None:
        max_frames = default_max
    if min_errors is None:
        min_errors = default_min

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, K_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload
            else:
                u = np.zeros(N, dtype=int)
                u[info_indices] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                frame_ok = np.array_equal(u_hat[info_indices], u[info_indices])
            else:
                frame_ok = np.array_equal(u_hat[info_indices], u[info_indices])

            if not frame_ok:
                num_errors += 1
                if crc_length > 0:
                    num_bit_errors += np.sum(
                        u_hat[info_indices][:K_info] != u[info_indices][:K_info]
                    )
                else:
                    num_bit_errors += np.sum(u_hat[info_indices] != u[info_indices])

            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / max(num_frames * K_info, 1)
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time*1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
