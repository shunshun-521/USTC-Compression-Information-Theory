"""
蒙特卡洛仿真主循环
"""
import numpy as np
import time

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
    info_indices: 信息位在 u 向量中的索引；若为 None 则取前 K 个非冻结假设（不推荐）。
    """
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
            if info_indices is not None:
                info_bits = rng.integers(0, 2, size=K_info, dtype=np.int8)
                u = np.zeros(N, dtype=np.int8)
                if crc_length > 0:
                    from decoder_scl import crc_encode

                    payload = crc_encode(info_bits, crc_length)
                    u[info_indices] = payload
                else:
                    u[info_indices] = info_bits
            else:
                info_bits = rng.integers(0, 2, size=K_info, dtype=np.int8)
                u = np.zeros(N, dtype=np.int8)
                u[:K] = info_bits if crc_length == 0 else 0

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += float(aux)

            if info_indices is not None:
                err_bits = np.sum(u_hat[info_indices][:K_info] != info_bits)
            else:
                err_bits = np.sum(u_hat[:K_info] != info_bits)

            if err_bits > 0:
                num_errors += 1
                num_bit_errors += err_bits
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames and K_info else 1.0
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
            extra = f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else ""
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms{extra}"
            )

    return results
