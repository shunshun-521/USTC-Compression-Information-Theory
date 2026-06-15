"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _permute_llr,
    f_operation,
    g_operation,
    sc_decode_recursive,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    mask = (1 << crc_length) - 1
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC_POLYNOMIALS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


def _partial_sum_up(u_segment):
    """计算分段的部分和向量（与递归 SC 中 u_hat_up 一致）"""
    u_segment = np.asarray(u_segment, dtype=np.int8)
    n = len(u_segment)
    if n == 1:
        return u_segment.copy()
    half = n // 2
    up_left = _partial_sum_up(u_segment[:half])
    up_right = _partial_sum_up(u_segment[half:])
    return np.concatenate([np.bitwise_xor(up_left, up_right), up_right])


def _leaf_llr_at_phi(llr_tree, frozen_bits, u_hat, phi):
    """已知 u_hat[0:phi] 时，计算第 phi 个比特的 LLR"""

    def helper(llr, frozen, bit_base):
        n = len(llr)
        if n == 1:
            return float(llr[0])

        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]

        if phi < bit_base + half:
            return helper(f_operation(llr1, llr2), frozen[:half], bit_base)

        u_seg = u_hat[bit_base : bit_base + half]
        u_up = _partial_sum_up(u_seg)
        x_llr2 = g_operation(llr1, llr2, u_up)
        return helper(x_llr2, frozen[half:], bit_base + half)

    return helper(llr_tree, frozen_bits, 0)


class SCLDecoder:
    """SCL 译码器（按相位迭代，Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_tree = _permute_llr(llr_ch)
        N = self.N
        paths = [(0.0, np.zeros(N, dtype=np.int8))]

        for phi in range(N):
            candidates = []
            for pm, u_hat in paths:
                llr_leaf = _leaf_llr_at_phi(llr_tree, self.frozen_bits, u_hat, phi)

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr_leaf >= 0 else abs(llr_leaf)
                    new_u = u_hat.copy()
                    new_u[phi] = 0
                    candidates.append((pm + penalty, new_u))
                else:
                    for bit in (0, 1):
                        penalty = (
                            0.0
                            if (bit == 0 and llr_leaf >= 0)
                            or (bit == 1 and llr_leaf < 0)
                            else abs(llr_leaf)
                        )
                        new_u = u_hat.copy()
                        new_u[phi] = bit
                        candidates.append((pm + penalty, new_u))

            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        best_idx = 0
        if self.crc_length > 0:
            valid = [
                i
                for i, (_, u) in enumerate(paths)
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                best_idx = min(valid, key=lambda i: paths[i][0])

        pm, u_hat = paths[best_idx]
        return u_hat.copy(), pm
