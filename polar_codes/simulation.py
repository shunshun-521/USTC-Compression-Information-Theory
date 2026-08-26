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
    info_indices=None,
    verbose=True,
    seed=42,
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
        info_indices: 信息位索引，用于 BER 统计
        verbose: 是否打印进度
        seed: 随机数种子

    返回：dict 列表，每个 Eb/N0 点包含仿真统计量
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    k_info = K - crc_length

    if info_indices is None:
        info_indices = np.arange(N)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, size=k_info, dtype=np.int8)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, size=K, dtype=np.int8)

            u = np.zeros(N, dtype=np.int8)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == 'bp':
                total_iters += aux

            if crc_length > 0:
                frame_error = not np.array_equal(u_hat[info_indices], payload)
                bit_errors = np.count_nonzero(u_hat[info_indices[:k_info]] != info_bits)
            else:
                frame_error = not np.array_equal(u_hat[info_indices], payload)
                bit_errors = np.count_nonzero(u_hat[info_indices] != payload)

            num_errors += int(frame_error)
            num_bit_errors += bit_errors
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
