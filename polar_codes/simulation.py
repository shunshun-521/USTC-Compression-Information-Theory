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
    verbose=True,
    seed=42,
    info_indices=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        raise ValueError("info_indices must be provided")

    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            info_bits = rng.integers(0, 2, size=K_info)

            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
            else:
                u[info_indices] = info_bits

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            frame_error = not np.array_equal(u_hat[info_indices], u[info_indices])
            if frame_error:
                num_errors += 1

            if crc_length > 0:
                bit_err = np.sum(u_hat[info_indices][:K_info] != info_bits)
            else:
                bit_err = np.sum(u_hat[info_indices] != info_bits)
            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info)
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


def get_sim_params():
    """从环境变量读取快速仿真参数"""
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not quick else "3000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not quick else "20"))
    return max_frames, min_errors


def get_eb_n0_range(default_range):
    """快速模式下使用稀疏 Eb/N0 采样"""
    if os.environ.get("POLAR_QUICK", "0") != "1":
        return default_range
    step = float(os.environ.get("POLAR_EB_N0_STEP", "0.5"))
    return np.arange(default_range[0], default_range[-1] + step * 0.5, step)
