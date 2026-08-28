"""
蒙特卡洛仿真主循环
"""
import numpy as np
import time

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_indices = np.arange(N)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)

            if crc_length > 0:
                info_bits = rng.integers(0, 2, K_info)
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
            else:
                u[info_indices] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            t1 = time.perf_counter()
            total_decode_time += t1 - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
                if decoder_type == "bp" and aux is not None:
                    total_iters += aux
            else:
                u_hat = decode_out

            if crc_length > 0:
                sent_info = info_bits
                recv_info = u_hat[info_indices][:K_info]
            else:
                sent_info = u[info_indices]
                recv_info = u_hat[info_indices]

            bit_err = np.sum(sent_info != recv_info)
            num_bit_errors += bit_err
            if bit_err > 0:
                num_errors += 1
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
