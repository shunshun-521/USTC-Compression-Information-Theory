"""
蒙特卡洛仿真主循环
"""
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
    """蒙特卡洛仿真。"""
    rng = np.random.default_rng(seed)
    rate = K / N
    if info_indices is None:
        info_indices = np.arange(N)
    info_indices = np.asarray(info_indices, dtype=int)
    K_info = len(info_indices) - crc_length if crc_length > 0 else len(info_indices)

    results = []

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
                info_payload = rng.integers(0, 2, K_info)
                coded_info = crc_encode(info_payload, crc_length)
                u[info_indices] = coded_info
            else:
                u[info_indices] = rng.integers(0, 2, len(info_indices))

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
                payload_hat = u_hat[info_indices][:-crc_length]
                payload_sent = u[info_indices][:-crc_length]
            else:
                payload_hat = u_hat[info_indices]
                payload_sent = u[info_indices]

            frame_err = not np.array_equal(payload_hat, payload_sent)
            num_bit_errors += np.count_nonzero(payload_hat != payload_sent)
            num_errors += int(frame_err)
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
                f"| AvgTime={avg_time*1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters else "")
            )

    return results
