"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


def _fast_sim_defaults():
    """快速仿真模式（环境变量 POLAR_FAST_SIM=1）"""
    if os.environ.get("POLAR_FAST_SIM", "0") == "1":
        return 500, 10
    return None, None


def fast_eb_n0_range(full_range):
    """快速模式下缩减 Eb/N0 采样点"""
    if os.environ.get("POLAR_FAST_SIM", "0") == "1":
        return full_range[::4]
    return full_range


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

    返回：dict 列表，每个 Eb/N0 点一条记录
    """
    fast_max, fast_min = _fast_sim_defaults()
    if fast_max is not None:
        max_frames = fast_max
        min_errors = fast_min

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_indices = np.where(np.arange(N) < K)[0]

    info_indices = np.asarray(info_indices, dtype=int)

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
                data_bits = rng.integers(0, 2, size=K_info)
                coded_info = crc_encode(data_bits, crc_length)
                u[info_indices] = coded_info
            else:
                u[info_indices] = rng.integers(0, 2, size=K)

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
                sent_data = u[info_indices][:K_info]
                recv_data = u_hat[info_indices][:K_info]
            else:
                sent_data = u[info_indices]
                recv_data = u_hat[info_indices]

            frame_error = not np.array_equal(sent_data, recv_data)
            if frame_error:
                num_errors += 1
            num_bit_errors += int(np.sum(sent_data != recv_data))
            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames > 0 else None

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
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
