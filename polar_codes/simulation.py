"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np
from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decoder,
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
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    k_info = K - crc_length

    if info_indices is None:
        raise ValueError("info_indices must be provided (GA 构造的信息位索引)")

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
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode_bits(info_bits, crc_length)
                u[info_indices] = payload
            else:
                u[info_indices] = rng.integers(0, 2, k_info)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = reorder_llr_for_decoder(compute_llr(y, sigma), N)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                rx_info = u_hat[info_indices]
                frame_err = not crc_check_bits(rx_info, crc_length)
                bit_err = np.sum(info_bits != u_hat[info_indices][:k_info])
            else:
                sent_info = u[info_indices]
                rx_info = u_hat[info_indices]
                frame_err = not np.array_equal(sent_info, rx_info)
                bit_err = np.sum(sent_info != rx_info)

            num_frames += 1
            num_bit_errors += bit_err
            if frame_err:
                num_errors += 1

        bler = num_errors / max(num_frames, 1)
        ber = num_bit_errors / max(num_frames * k_info, 1)
        avg_time = total_decode_time / max(num_frames, 1)
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

        result = {
            "eb_n0_db": float(eb_n0_db),
            "bler": float(bler),
            "ber": float(ber),
            "num_errors": int(num_errors),
            "num_frames": int(num_frames),
            "avg_decode_time": float(avg_time),
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


def crc_encode_bits(info_bits, crc_length):
    from decoder_scl import crc_encode
    return crc_encode(info_bits, crc_length)


def crc_check_bits(bits, crc_length):
    from decoder_scl import crc_check
    return crc_check(bits, crc_length)
