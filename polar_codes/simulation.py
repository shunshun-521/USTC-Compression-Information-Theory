"""
蒙特卡洛仿真主循环
"""
import numpy as np
import time
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


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

    返回每个 Eb/N0 点的结果 dict 列表。
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
                info_raw = rng.integers(0, 2, k_info)
                info_with_crc = crc_encode(info_raw, crc_length)
                u_sent = np.zeros(N, dtype=int)
                u_sent[info_indices] = info_with_crc
            else:
                u_sent = np.zeros(N, dtype=int)
                u_sent[info_indices] = rng.integers(0, 2, K)

            x = polar_encode(u_sent)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            elapsed = time.perf_counter() - t0
            total_decode_time += elapsed

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                compare_idx = info_indices[:k_info]
            else:
                compare_idx = info_indices

            frame_error = not np.array_equal(u_sent[compare_idx], u_hat[compare_idx])
            if frame_error:
                num_errors += 1
            num_bit_errors += np.sum(u_sent[compare_idx] != u_hat[compare_idx])
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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
