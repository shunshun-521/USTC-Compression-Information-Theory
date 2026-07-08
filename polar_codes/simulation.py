"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import bit_reversal_permutation, polar_encode
from utils import crc_check, crc_encode


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
    SC/SCL 使用比特倒序 LLR；BP 使用自然序信道 LLR。
    """
    from construction import ga_construction

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, 2.5)

    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=K_info)
                info_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_with_crc
            else:
                info_payload = rng.integers(0, 2, size=K)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr_raw = compute_llr(y, sigma)
            if decoder_type == "bp":
                llr = llr_raw
            else:
                llr = llr_raw[bit_reversal_permutation(N)]

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices]
                frame_err = not crc_check(decoded_info, crc_length)
                bit_err = np.sum(decoded_info[:K_info] != info_payload)
            else:
                decoded_info = u_hat[info_indices]
                bit_err = np.sum(decoded_info != info_payload)
                frame_err = bit_err > 0

            num_bit_errors += bit_err
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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time*1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
