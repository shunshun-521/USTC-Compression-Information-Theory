"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _cn_op, _prepare_llr, _vn_op, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, _crc_remainder(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], _crc_remainder(info, crc_length))


class SCLDecoder:
    """SCL 译码器（递归列表，与 SC 算法对齐）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen == 0)[0]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen), 0.0

        llr = _prepare_llr(llr_ch)

        def expand(paths, llr_node, fr):
            n = len(llr_node)
            if n == 1:
                out = []
                for pm, u_hat in paths:
                    if fr[0]:
                        out.append((pm + self._penalty(llr_node[0], 0), np.concatenate([u_hat, [0]])))
                    else:
                        for bit in (0, 1):
                            out.append(
                                (pm + self._penalty(llr_node[0], bit), np.concatenate([u_hat, [bit]]))
                            )
                out.sort(key=lambda x: x[0])
                return out[: self.list_size]

            half = n // 2
            l1, l2 = llr_node[:half], llr_node[half:]
            left_paths = expand(paths, _cn_op(l1, l2), fr[:half])
            out = []
            for pm, u_left in left_paths:
                g_in = _vn_op(l1, l2, u_left)
                right_paths = expand([(pm, np.array([], dtype=int))], g_in, fr[half:])
                for pm_r, u_right in right_paths:
                    u_full = np.concatenate([u_left, u_right])
                    out.append((pm_r, u_full))
            out.sort(key=lambda x: x[0])
            return out[: self.list_size]

        paths = expand([(0.0, np.array([], dtype=int))], llr, self.frozen)
        candidates = paths

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in candidates:
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                candidates = valid

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u.astype(int), best_pm
