"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
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
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if frozen_bits is None:
        raise ValueError("frozen_bits must be provided")

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if info_indices is None:
        info_indices = np.where(~frozen_bits)[0]
    else:
        info_indices = np.asarray(info_indices, dtype=int)

    k_info = K - crc_length

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

                info = rng.integers(0, 2, size=k_info)
                payload = crc_encode(info, crc_length)
                u[info_indices] = payload
            else:
                info = rng.integers(0, 2, size=k_info)
                u[info_indices] = info

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
                frame_err = not np.array_equal(
                    u_hat[info_indices][:k_info], info
                )
            else:
                frame_err = not np.array_equal(u_hat[info_indices], info)

            if frame_err:
                num_errors += 1
                if crc_length > 0:
                    num_bit_errors += np.sum(u_hat[info_indices][:k_info] != info)
                else:
                    num_bit_errors += np.sum(u_hat[info_indices] != info)

            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info)
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
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters else "")
            )

    return results


def fast_sim_defaults():
    """快速仿真参数（环境变量 POLAR_FAST_SIM=1 时启用）"""
    if os.environ.get("POLAR_FAST_SIM", "0") == "1":
        return {"max_frames": 5000, "min_errors": 20}
    return {"max_frames": 100000, "min_errors": 100}
