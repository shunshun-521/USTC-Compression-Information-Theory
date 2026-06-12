"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
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

    参数：
        N: 码长
        K: 信息位数（含 CRC 长度的有效信息位）
        eb_n0_db_list: Eb/N0 列表（dB）
        decoder: 译码器对象或函数
        decoder_type: 译码器类型标识
        max_frames: 每个信噪比点最大仿真帧数
        min_errors: 每个信噪比点最少错误帧数
        crc_length: CRC 长度（用于 CA-SCL）
        verbose: 是否打印进度
        seed: 随机数种子
        info_indices: 信息位索引（可选，默认 GA 构造）

    返回：dict 列表，每个 Eb/N0 点一条记录
    """
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", max_frames))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", min_errors))

    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, 2.5)
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
            info_bits = rng.integers(0, 2, K_info)
            if crc_length > 0:
                u[info_indices] = crc_encode(info_bits, crc_length)
            else:
                u[info_indices] = info_bits

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
            else:
                u_hat, aux = decode_out, None

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            u_info_sent = info_bits
            u_info_hat = u_hat[info_indices][:K_info]
            bit_err = np.sum(u_info_sent != u_info_hat)
            frame_err = int(bit_err > 0)

            num_bit_errors += bit_err
            num_errors += frame_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
        avg_time = total_decode_time / num_frames
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
