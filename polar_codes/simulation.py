"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
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
    verbose=True,
    seed=42,
    info_indices=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
    else:
        info_idx = np.asarray(info_indices)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, size=K_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_idx] = payload
            else:
                info_bits = rng.integers(0, 2, size=K_info)
                u = np.zeros(N, dtype=int)
                u[info_idx] = info_bits

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            decoded_info = u_hat[info_idx][:K_info]
            frame_error = not np.array_equal(decoded_info, info_bits)
            bit_errors = np.sum(decoded_info != info_bits)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_errors

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


def get_fast_sim_params():
    """根据环境变量返回快速/完整仿真参数"""
    if os.environ.get("POLAR_FAST_SIM", "0") == "1":
        return {"max_frames": 2000, "min_errors": 20}
    return {"max_frames": 100000, "min_errors": 100}
