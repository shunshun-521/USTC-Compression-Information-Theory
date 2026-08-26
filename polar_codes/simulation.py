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
)


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

    参数：
        info_indices: 信息位索引；若为 None 则假定低可靠性位为冻结位（不推荐）
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices = np.sort(np.argsort(np.arange(N))[-K:])
    info_indices = np.asarray(info_indices, dtype=int)
    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)

            if crc_length > 0:
                info_raw = rng.integers(0, 2, size=K_info)
                info_with_crc = crc_encode_from_sim(info_raw, crc_length)
                u[info_indices] = info_with_crc
            else:
                info_bits = rng.integers(0, 2, size=K)
                u[info_indices] = info_bits

            from encoder import polar_encode

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            sent_info = u[info_indices]
            if crc_length > 0:
                sent_info = sent_info[:K_info]
                recv_info = u_hat[info_indices][:K_info]
            else:
                recv_info = u_hat[info_indices]

            bit_err = int(np.sum(sent_info != recv_info))
            num_bit_errors += bit_err
            if bit_err > 0:
                num_errors += 1
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info)
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
            extra = f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else ""
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms{extra}"
            )

    return results


def crc_encode_from_sim(info_bits, crc_length):
    from decoder_scl import crc_encode

    return crc_encode(info_bits, crc_length)
