"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.asarray(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


def _path_metric(llr_bit, u):
    hard = 0 if llr_bit >= 0 else 1
    return 0.0 if u == hard else abs(llr_bit)


def _segment_up(u_segment):
    """计算子树 g 运算所需的 stage 部分和"""
    n = len(u_segment)
    if n == 1:
        return u_segment.copy()
    half = n // 2
    left_up = _segment_up(u_segment[:half])
    right_up = _segment_up(u_segment[half:])
    return np.concatenate([np.bitwise_xor(left_up, right_up), right_up])


class SCLDecoder:
    """SCL 译码器（递归列表分裂）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length

    def _scl_recursive(self, llr_node, frozen_sub, bit_offset, paths):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            out = []
            for pm, u_hat in paths:
                llr_bit = llr_node[0]
                if frozen_sub[0]:
                    u_hat_c = u_hat.copy()
                    u_hat_c[idx] = 0
                    out.append((pm + _path_metric(llr_bit, 0), u_hat_c))
                else:
                    for u in (0, 1):
                        u_hat_c = u_hat.copy()
                        u_hat_c[idx] = u
                        out.append((pm + _path_metric(llr_bit, u), u_hat_c))
            out.sort(key=lambda x: x[0])
            return out[: self.L]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        paths_left = self._scl_recursive(llr_left, frozen_sub[:half], bit_offset, paths)
        paths_left = sorted(paths_left, key=lambda x: x[0])[: self.L]

        merged = []
        for pm, u_hat in paths_left:
            u1 = u_hat[bit_offset : bit_offset + half]
            u1_up = _segment_up(u1)
            llr_right = g_operation(llr_node[:half], llr_node[half:], u1_up)
            paths_right = self._scl_recursive(
                llr_right, frozen_sub[half:], bit_offset + half, [(pm, u_hat)]
            )
            merged.extend(paths_right)

        merged.sort(key=lambda x: x[0])
        return merged[: self.L]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u0 = np.zeros(self.N, dtype=int)
        paths = self._scl_recursive(llr_ch, self.frozen_bits, 0, [(0.0, u0)])
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for pm, u_hat in sorted(paths, key=lambda x: x[0]):
                payload = u_hat[info_positions]
                if crc_check(payload, self.crc_length):
                    return u_hat.copy(), pm

        best = min(paths, key=lambda x: x[0])
        return best[1].copy(), best[0]
