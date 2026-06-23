"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
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

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if info_indices is None:
                info_idx = np.where(np.arange(N) < K)[0]
            else:
                info_idx = info_indices

            info_bits = rng.integers(0, 2, size=k_info)
            u = np.zeros(N, dtype=int)

            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u[info_idx[: len(payload)]] = payload
            else:
                u[info_idx[:k_info]] = info_bits

            x = polar_encode(u)
            y = bpsk_modulate(x)
            noise = rng.normal(0.0, sigma, size=N)
            llr = compute_llr(y + noise, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
                if decoder_type == "bp" and aux is not None:
                    total_iters += aux
            else:
                u_hat = decode_out

            num_frames += 1

            if crc_length > 0:
                decoded_info = u_hat[info_idx[: k_info + crc_length]]
                frame_ok = np.array_equal(info_bits, decoded_info[:k_info])
            else:
                frame_ok = np.array_equal(info_bits, u_hat[info_idx[:k_info]])

            if not frame_ok:
                num_errors += 1
                num_bit_errors += np.sum(info_bits != u_hat[info_idx[:k_info]])

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
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
