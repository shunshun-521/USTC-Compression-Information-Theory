"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import cn_operation, g_operation, sc_decode


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _partial_up_from_u(u_seg):
    """由已知 u 段计算 g 运算所需的部分和向量。"""
    u_seg = np.asarray(u_seg, dtype=int)
    if len(u_seg) == 1:
        return u_seg.astype(np.float64)
    half = len(u_seg) // 2
    up_left = _partial_up_from_u(u_seg[:half])
    up_right = _partial_up_from_u(u_seg[half:])
    u_up_l = np.bitwise_xor(up_left.astype(int), up_right.astype(int)).astype(np.float64)
    return np.concatenate([u_up_l, up_right])


def _root_llr_at_phi(llr, u_prefix, phi):
    """计算比特 phi 处的根 LLR（已知 u_prefix[0:phi]）。"""

    def walk(node, offset):
        n = len(node)
        if n == 1:
            return node[0]
        half = n // 2
        top = cn_operation(node[:half], node[half:])
        if phi < offset + half:
            return walk(top, offset)
        u_left = u_prefix[offset : offset + half]
        u_left_up = _partial_up_from_u(u_left)
        bottom = g_operation(node[:half], node[half:], u_left_up)
        return walk(bottom, offset + half)

    return walk(np.asarray(llr, dtype=np.float64), 0)


def _path_penalty(llr_val, bit):
    return 0.0 if bit == (0 if llr_val >= 0 else 1) else abs(llr_val)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, L = self.N, self.list_size

        if L == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [(0.0, np.zeros(N, dtype=int))]

        for phi in range(N):
            new_paths = []
            for pm, u in paths:
                llr_phi = _root_llr_at_phi(llr_ch, u, phi)
                if self.frozen_bits[phi]:
                    u_new = u.copy()
                    u_new[phi] = 0
                    new_paths.append((pm + _path_penalty(llr_phi, 0), u_new))
                else:
                    for bit in (0, 1):
                        u_new = u.copy()
                        u_new[phi] = bit
                        new_paths.append((pm + _path_penalty(llr_phi, bit), u_new))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[:L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p[1][self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[0])
        return best[1], best[0]
