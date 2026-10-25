"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode, bit_reversal_permutation
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
    info_indices=None,
    frozen_bits=None,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。
    frozen_bits: 长度 N，1=冻结，0=信息
    info_indices: 信息位索引（用于 BER 统计）
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    br_perm = bit_reversal_permutation(N)
    if frozen_bits is None:
        raise ValueError("frozen_bits required")
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    if info_indices is None:
        info_indices = np.where(~frozen_bits)[0]
    info_indices = np.asarray(info_indices)
    K_info = len(info_indices) - crc_length

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
                info_payload = rng.integers(0, 2, size=K_info, dtype=np.int8)
                coded = crc_encode(info_payload, crc_length)
                u[info_indices] = coded
            else:
                u[info_indices] = rng.integers(0, 2, size=K, dtype=np.int8)

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)[br_perm]

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                payload_hat = u_hat[info_indices][:K_info]
                payload_sent = u[info_indices][:K_info]
                frame_err = not np.array_equal(payload_hat, payload_sent)
                bit_err = np.sum(payload_hat != payload_sent)
            else:
                payload_hat = u_hat[info_indices]
                payload_sent = u[info_indices]
                frame_err = not np.array_equal(payload_hat, payload_sent)
                bit_err = np.sum(payload_hat != payload_sent)

            num_errors += int(frame_err)
            num_bit_errors += int(bit_err)
            num_frames += 1

        bler = num_errors / num_frames
        ber_den = num_frames * max(K_info, 1)
        ber = num_bit_errors / ber_den
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
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
