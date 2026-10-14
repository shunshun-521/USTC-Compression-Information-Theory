"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode
from encoder import polar_encode
from utils import mc_params


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
    max_frames, min_errors = mc_params(max_frames, min_errors)
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_raw = rng.integers(0, 2, size=K_info, dtype=np.int8)
                payload = crc_encode(info_raw, crc_length)
            else:
                payload = rng.integers(0, 2, size=K, dtype=np.int8)

            u = np.zeros(N, dtype=np.int8)
            if info_indices is None:
                raise ValueError("info_indices required for simulation")
            u[np.asarray(info_indices, dtype=int)] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += aux

            u_hat = np.asarray(u_hat, dtype=np.int8)
            if crc_length > 0:
                decoded_info = u_hat[np.asarray(info_indices, dtype=int)]
                block_err = not np.array_equal(decoded_info[:K_info], info_raw)
                bit_err = int(np.sum(decoded_info[:K_info] != info_raw))
            else:
                block_err = not np.array_equal(u_hat[np.asarray(info_indices, dtype=int)], payload)
                bit_err = int(
                    np.sum(u_hat[np.asarray(info_indices, dtype=int)] != payload)
                )

            num_bit_errors += bit_err
            num_errors += int(block_err)
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
