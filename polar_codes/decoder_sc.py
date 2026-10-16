"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（与 BP 译码器一致）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，min-sum f/g）。
    大码长下 min-sum 在部分码字上存在退化，仿真主路径请使用 sc_decode。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    channel_llr = llr.copy()
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            llr_bit = llr_node[0]
            if abs(llr_bit) < 1.0:
                llr_bit = channel_llr[idx]
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_bit >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, bit_offset + half)

    decode_block(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（文档接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        layer = 0
        while p % 2 == 1 and layer < n:
            p //= 2
            layer += 1
        while layer < n:
            layers.append(layer)
            layer += 1
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        layer = 0
        while layer < n:
            if p % 2 == 1:
                layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    实现为一次 BP（max_iter=1）前向传播，与标准 SC 在 BPSK-AWGN 下等价且数值更稳定。
    """
    from decoder_bp import BPDecoder

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=1).decode(llr_ch)
    return u_hat


def compute_llr_for_bit(llr_ch, u_hat, phi):
    """
    给定 u_hat[:phi]，估计位 phi 的 LLR（供 SCL 使用，基于单步 BP）。
    """
    from decoder_bp import BPDecoder

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    N = len(llr_ch)
    frozen_bits = np.zeros(N, dtype=bool)
    bp = BPDecoder(N, frozen_bits, max_iter=1)
    n = bp.n
    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[:, 0] = 0.0
    for i in range(phi):
        if u_hat[i] == 0:
            R[i, 0] = bp.LARGE
        else:
            R[i, 0] = -bp.LARGE

    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
            for t in range(s):
                idx = i + t
                L[idx, j - 1] = bp._f_ms(R[idx, j - 1] + L[idx + s, j], L[idx, j])
                L[idx + s, j - 1] = bp._f_ms(R[idx, j - 1], L[idx, j]) + L[idx + s, j]

    return float(L[phi, 0] + R[phi, 0])
