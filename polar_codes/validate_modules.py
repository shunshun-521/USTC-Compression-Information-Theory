"""模块数值校验（各实验脚本导入前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def _build_GN(N):
    """生成矩阵 G_N = F^{\\otimes n}（与蝶形编码一致）。"""
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


def validate_encoder():
    N = 4
    GN = _build_GN(N)
    for _ in range(20):
        u = np.random.randint(0, 2, N)
        x_ref = (u @ GN) % 2
        x = polar_encode(u)
        assert np.array_equal(x, x_ref), f"编码器错误: u={u}, x={x}, ref={x_ref}"
    # 规范示例：u=[1,0,1,1]（与 stride-first 蝶形编码一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, (u @ _build_GN(4)) % 2), f"编码器规范示例错误: {x}"
    print("  [OK] 编码器校验通过")


def validate_sc():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = 0.05  # 极高 SNR（近似无噪）
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧错误"
    print("  [OK] SC 译码校验通过")


def validate_scl_equiv_sc():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(6.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        u_ref = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_sc, u_ref), "L=1 SCL 与 SC 不一致"
        assert np.array_equal(u_scl, u_ref)
    print("  [OK] SCL(L=1) 与 SC 等价校验通过")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert not crc_check(enc[:-1], 8)
    print("  [OK] CRC 校验通过")


def run_all_validations():
    print("运行模块校验...")
    validate_encoder()
    validate_sc()
    validate_scl_equiv_sc()
    validate_crc()
    print("全部校验通过。\n")


if __name__ == "__main__":
    run_all_validations()
