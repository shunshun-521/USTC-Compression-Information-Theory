"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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
    design_eb_n0_db=2.5,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。

    参数：
        N: 码长
        K: 信息位数（含 CRC 时 K 为编码后信息位总数）
        eb_n0_db_list: Eb/N0 列表（dB）
        decoder: 译码器 callable(llr_ch) -> (u_hat, aux)
        decoder_type: 'sc', 'scl', 'bp'
        max_frames: 每个信噪比点最大仿真帧数
        min_errors: 每个信噪比点最少错误帧数
        crc_length: CRC 长度（CA-SCL）
        design_eb_n0_db: GA 构造设计信噪比
        verbose: 是否打印进度
        seed: 随机数种子

    返回：每个 Eb/N0 点的结果 dict 列表
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    info_idx, frozen_idx, _ = ga_construction(N, K, design_eb_n0_db, rate=rate)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

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
            if crc_length > 0:
                info_bits = rng.integers(0, 2, K_info)
                payload = crc_encode(info_bits, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_idx] = payload
            else:
                u = np.zeros(N, dtype=int)
                u[info_idx] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                frame_err = not np.array_equal(u_hat[info_idx], u[info_idx])
                bit_err = np.sum(u_hat[info_idx] != u[info_idx])
            else:
                frame_err = not np.array_equal(u_hat[info_idx], u[info_idx])
                bit_err = np.sum(u_hat[info_idx] != u[info_idx])

            num_frames += 1
            if frame_err:
                num_errors += 1
            num_bit_errors += bit_err

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames and K_info else 0.0
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
