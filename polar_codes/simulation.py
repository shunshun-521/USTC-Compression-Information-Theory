"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    prepare_channel_llr,
)
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

    返回每个 Eb/N0 点的结果字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        raise ValueError("info_indices must be provided")
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
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, size=K_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload
                u_sent_info = info_bits
            else:
                u = np.zeros(N, dtype=int)
                u[info_indices] = rng.integers(0, 2, size=K)
                u_sent_info = u[info_indices]

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = prepare_channel_llr(compute_llr(y, sigma))

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if not np.array_equal(u_hat[info_indices][:K_info], u_sent_info):
                num_errors += 1
                num_bit_errors += np.sum(u_hat[info_indices][:K_info] != u_sent_info)

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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
