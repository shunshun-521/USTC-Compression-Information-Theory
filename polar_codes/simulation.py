"""
蒙特卡洛仿真主循环
"""
import numpy as np
import time

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode, channel_llr_to_decoder, polar_encode_natural
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
    use_natural_llr=False,
):
    """
    蒙特卡洛仿真。

    参数：
        info_indices: 信息位索引；若 None 则统计全部 K 个非冻结位
        use_natural_llr: True 时不对 LLR 做比特倒序（BP 译码）
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
            if crc_length > 0:
                info_raw = rng.integers(0, 2, size=k_info)
                payload = crc_encode(info_raw, crc_length)
                u = np.zeros(N, dtype=int)
                if info_indices is None:
                    raise ValueError("info_indices required when crc_length > 0")
                u[info_indices] = payload
            else:
                u = np.zeros(N, dtype=int)
                if info_indices is None:
                    info_positions = None
                    u[:K] = rng.integers(0, 2, size=K)
                else:
                    info_positions = info_indices
                    u[info_positions] = rng.integers(0, 2, size=K)

            if use_natural_llr:
                x = polar_encode_natural(u)
            else:
                x = polar_encode(u)

            y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
            llr = compute_llr(y, sigma)

            if not use_natural_llr:
                llr = channel_llr_to_decoder(llr)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
                if decoder_type == "bp" and aux is not None:
                    total_iters += aux
            else:
                u_hat = decode_out

            if crc_length > 0:
                sent_info = payload
                recv_info = u_hat[info_indices]
            elif info_indices is not None:
                sent_info = u[info_indices]
                recv_info = u_hat[info_indices]
            else:
                sent_info = u[:K]
                recv_info = u_hat[:K]

            bit_err = int(np.sum(sent_info != recv_info))
            num_bit_errors += bit_err
            if bit_err > 0:
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
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
