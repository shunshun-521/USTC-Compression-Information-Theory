"""
蒙特卡洛仿真主循环
"""
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
):
    """
    蒙特卡洛仿真。
    """
    from construction import ga_construction
    from decoder_scl import crc_encode

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
    else:
        info_idx = np.asarray(info_indices)

    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=np.int64)
            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=K_info)
                payload = crc_encode(info_bits, crc_length)
                u[info_idx[: len(payload)]] = payload
            else:
                u[info_idx] = rng.integers(0, 2, size=K)

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
                sent_info = u[info_idx[:K_info]]
                recv_info = u_hat[info_idx[:K_info]]
            else:
                sent_info = u[info_idx]
                recv_info = u_hat[info_idx]

            bit_err = np.sum(sent_info != recv_info)
            if bit_err > 0:
                num_errors += 1
                num_bit_errors += bit_err

            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = (
            num_bit_errors / (num_frames * K_info) if num_frames > 0 and K_info > 0 else 0.0
        )
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
        avg_iters = (
            (total_iters / num_frames) if decoder_type == "bp" and num_frames > 0 else None
        )

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
