"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
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
    verbose=True,
    seed=42,
    design_eb_n0_db=2.5,
    info_indices=None,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)

    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_indices] = 0

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            info_bits = rng.integers(0, 2, k_info)

            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u[info_indices[: len(payload)]] = payload
            else:
                u[info_indices] = info_bits

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
                sent_payload = u[info_indices[:K]]
                recv_payload = u_hat[info_indices[:K]]
                frame_error = not np.array_equal(recv_payload, sent_payload)
                bit_err = np.sum(recv_payload[:k_info] != info_bits)
            else:
                frame_error = not np.array_equal(u_hat[info_indices], info_bits)
                bit_err = np.sum(u_hat[info_indices] != info_bits)

            num_frames += 1
            if frame_error:
                num_errors += 1
            num_bit_errors += bit_err

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames else 1.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

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
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results


def get_sim_params():
    """读取环境变量以支持快速/完整仿真切换"""
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "500" if quick else "100000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "10" if quick else "100"))
    return max_frames, min_errors
