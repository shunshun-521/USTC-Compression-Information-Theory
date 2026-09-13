"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode
from encoder import polar_encode


def _sim_params():
    """支持通过环境变量加速仿真（自动化/调试）。"""
    fast = os.environ.get("POLAR_FAST_SIM", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not fast else "5000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not fast else "20"))
    stride = int(os.environ.get("POLAR_SNR_STRIDE", "1" if not fast else "2"))
    return max_frames, min_errors, stride


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    info_indices,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。

    参数：
        info_indices: 信息位在 u 向量中的索引
    """
    default_max, default_min, stride = _sim_params()
    if max_frames is None:
        max_frames = default_max
    if min_errors is None:
        min_errors = default_min

    if stride > 1:
        eb_n0_db_list = eb_n0_db_list[::stride]

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    k_info = K - crc_length
    info_indices = np.asarray(info_indices, dtype=int)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=k_info)
                payload = crc_encode(info_bits, crc_length)
            else:
                info_bits = rng.integers(0, 2, size=K)
                payload = info_bits

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload

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
            frame_error = not np.array_equal(decoded_info[:k_info], info_bits)
            if frame_error:
                num_errors += 1
            num_bit_errors += np.sum(decoded_info[:k_info] != info_bits)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
