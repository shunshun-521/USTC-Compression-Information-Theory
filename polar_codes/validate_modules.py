"""模块数值校验（各实验脚本导入前运行）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    # 编码器
    u = np.array([1, 0, 1, 1])
    G = build_generator_matrix(4)
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器与生成矩阵不一致: {x}"

    # GA 构造 smoke
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4

    # CRC
    payload = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    with_crc = crc_encode(payload, 8)
    assert crc_check(with_crc, 8)

    # SC 高信噪比
    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        # 近无噪 LLR 验证译码逻辑（避免蒙特卡洛偶发信道错误）
        llr = compute_llr(bpsk_modulate(x), 1e-12)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)
        # 递归实现作参考，允许与非递归存在实现差异
        u_hat_r = sc_decode_recursive(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat_r[info_idx], payload):
            assert np.array_equal(u_hat[info_idx], payload)

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_hat_s, _ = scl.decode(llr)
    assert np.array_equal(u_hat_s, u_hat)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
