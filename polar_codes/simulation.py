"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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

    decoder: 可调用对象 decoder(llr) -> (u_hat, aux)
    info_indices: 信息位索引（用于统计 BER）；若为 None 则假定前 K 个非冻结位由调用方处理。
    """
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", max_frames))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", min_errors))

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
            u = np.zeros(N, dtype=np.int8)
            if crc_length > 0:
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, size=k_info, dtype=np.int8)
                coded = crc_encode(info_bits, crc_length)
                u[np.array(info_indices, dtype=int)] = coded
            else:
                if info_indices is None:
                    raise ValueError("info_indices required when crc_length=0")
                info_bits = rng.integers(0, 2, size=K, dtype=np.int8)
                u[np.array(info_indices, dtype=int)] = info_bits

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += float(aux)

            idx = np.array(info_indices, dtype=int)
            if crc_length > 0:
                cmp_len = k_info
                err_bits = int(np.sum(u_hat[idx][:cmp_len] != u[idx][:cmp_len]))
            else:
                cmp_len = K
                err_bits = int(np.sum(u_hat[idx] != u[idx]))

            num_bit_errors += err_bits
            if err_bits > 0:
                num_errors += 1
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
