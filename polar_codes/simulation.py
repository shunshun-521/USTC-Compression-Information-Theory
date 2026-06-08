"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

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
    verbose=True,
    seed=42,
    info_indices=None,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    if info_indices is None:
        raise ValueError("info_indices must be provided")

    info_indices = np.asarray(info_indices, dtype=int)
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length
    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            info_bits = rng.integers(0, 2, k_info)
            u = np.zeros(N, dtype=int)

            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u[info_indices] = payload
                check_idx = info_indices
            else:
                u[info_indices] = info_bits
                check_idx = info_indices

            codeword = polar_encode(u)
            symbols = bpsk_modulate(codeword)
            received = awgn_channel(symbols, sigma, rng)
            llr = compute_llr(received, sigma)

            t0 = time.perf_counter()
            dec_out = decoder(llr)
            t1 = time.perf_counter()

            if isinstance(dec_out, tuple):
                u_hat, aux = dec_out
                if aux is not None and decoder_type == "bp":
                    total_iters += float(aux)
            else:
                u_hat = dec_out

            total_decode_time += t1 - t0
            num_frames += 1

            if not np.array_equal(u_hat[check_idx], u[check_idx]):
                num_errors += 1
                num_bit_errors += int(np.sum(u_hat[check_idx] != u[check_idx]))

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

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
