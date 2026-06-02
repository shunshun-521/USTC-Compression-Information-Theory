"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from construction import ga_construction


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
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。

    参数：
        info_indices: 信息位索引（可选，若提供则跳过 GA 构造）
        frozen_bits: 冻结位标记数组（1=冻结，可选）
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length

    if frozen_bits is None:
        if info_indices is None:
            info_indices, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0
    frozen_bool = frozen_bits.astype(bool)
    if info_indices is None:
        info_indices = np.where(~frozen_bool)[0]

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                msg = rng.integers(0, 2, k_info)
                from decoder_scl import crc_encode

                payload = crc_encode(msg, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload
            else:
                u = np.zeros(N, dtype=int)
                u[info_indices] = rng.integers(0, 2, K)

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
                check_bits = u_hat[info_indices]
                frame_ok = crc_check(check_bits, crc_length)
                if not frame_ok:
                    num_errors += 1
                bit_err = np.sum(u[info_indices[:k_info]] != u_hat[info_indices[:k_info]])
            else:
                frame_ok = np.array_equal(u[info_indices], u_hat[info_indices])
                if not frame_ok:
                    num_errors += 1
                bit_err = np.sum(u[info_indices] != u_hat[info_indices])

            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

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


def crc_check(bits, crc_length):
    from decoder_scl import crc_check as _crc_check

    return _crc_check(bits, crc_length)
