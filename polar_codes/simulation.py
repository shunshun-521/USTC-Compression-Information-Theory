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

    返回每个 Eb/N0 点的统计字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices = np.arange(N)

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=k_info)
                source_info = crc_encode(info_payload, crc_length)
            else:
                source_info = rng.integers(0, 2, size=K)

            u = np.zeros(N, dtype=int)
            u[info_indices] = source_info

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += aux

            decoded_info = u_hat[info_indices]
            if crc_length > 0:
                frame_err = not np.array_equal(decoded_info[:k_info], info_payload)
                bit_err = np.sum(decoded_info[:k_info] != info_payload)
            else:
                frame_err = not np.array_equal(decoded_info, source_info)
                bit_err = np.sum(decoded_info != source_info)

            num_errors += int(frame_err)
            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results


def get_sim_params(default_max_frames=100000, default_min_errors=100):
    """从环境变量读取快速仿真参数。"""
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", default_max_frames))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", default_min_errors))
    if quick:
        max_frames = min(max_frames, 2000)
        min_errors = min(min_errors, 15)
    return max_frames, min_errors


def get_eb_n0_range(start, stop, step):
    """仿真 Eb/N0 网格；POLAR_QUICK=1 时使用更稀疏网格。"""
    if os.environ.get("POLAR_QUICK", "0") == "1":
        return np.arange(start, stop, max(step, 0.5))
    return np.arange(start, stop, step)


def get_n_list(default_list):
    """POLAR_QUICK=1 时缩短码长列表。"""
    if os.environ.get("POLAR_QUICK", "0") == "1":
        return [default_list[0]]
    return default_list
