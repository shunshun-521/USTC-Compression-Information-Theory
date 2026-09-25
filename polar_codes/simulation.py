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
    decoder_type='sc',
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
):
    """
    蒙特卡洛仿真。

    返回：每个 Eb/N0 点的结果 dict 列表
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if info_indices is None:
        info_indices = np.where(np.ones(N, dtype=int) == 0)[0]

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
                payload = rng.integers(0, 2, size=K_info)
                coded_info = crc_encode(payload, crc_length)
                u[info_indices] = coded_info
            else:
                u[info_indices] = rng.integers(0, 2, size=K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == 'bp':
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices]
                frame_ok = crc_check_safe(decoded_info, crc_length)
                bit_err = np.sum(decoded_info[:K_info] != payload) if frame_ok else K_info
            else:
                bit_err = np.sum(u_hat[info_indices] != u[info_indices])
                frame_ok = bit_err == 0

            num_bit_errors += bit_err
            num_errors += 0 if frame_ok else 1
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info)
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' else None

        result = {
            'eb_n0_db': float(eb_n0_db),
            'bler': float(bler),
            'ber': float(ber),
            'num_errors': int(num_errors),
            'num_frames': int(num_frames),
            'avg_decode_time': float(avg_time),
            'avg_iters': avg_iters,
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


def crc_check_safe(bits, crc_length):
    from decoder_scl import crc_check
    if len(bits) < crc_length:
        return False
    return crc_check(bits, crc_length)
