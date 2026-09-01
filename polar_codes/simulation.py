"""
蒙特卡洛仿真主循环
"""
import os
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
        N: 码长
        K: 信息位数（含 CRC 长度的有效信息位）
        eb_n0_db_list: Eb/N0 列表（dB）
        decoder: 译码器对象或函数
        decoder_type: 译码器类型标识
        max_frames: 每个信噪比点最大仿真帧数
        min_errors: 每个信噪比点最少错误帧数
        crc_length: CRC 长度（用于 CA-SCL）
        info_indices: 信息位索引数组
        verbose: 是否打印进度
        seed: 随机数种子

    返回：
        dict 列表，每个 Eb/N0 点包含仿真统计量
    """
    if os.environ.get("POLAR_FAST_SIM") == "1":
        max_frames = min(max_frames, 5000)
        min_errors = min(min_errors, 20)

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length

    if info_indices is None:
        info_indices = np.arange(N)

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            # 生成信息比特
            info_bits = rng.integers(0, 2, size=K_info)
            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = info_bits

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            decoded_info = u_hat[info_indices]
            if crc_length > 0:
                frame_error = not np.array_equal(decoded_info, payload)
            else:
                frame_error = not np.array_equal(decoded_info, info_bits)

            if frame_error:
                num_errors += 1
                num_bit_errors += np.sum(decoded_info[:K_info] != info_bits)

            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
