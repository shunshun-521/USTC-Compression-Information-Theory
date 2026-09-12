"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode, f_operation, _frozen_to_set


CRC8_DIVISOR = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
CRC16_DIVISOR = np.array([int(b) for b in format(0x8005, '016b')], dtype=int)


def _crc_divide(bits, divisor):
    bits = np.asarray(bits, dtype=int).copy()
    r = len(divisor) - 1
    for i in range(len(bits) - r):
        if bits[i] == 1:
            bits[i:i + len(divisor)] ^= divisor
    return bits[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    divisor = CRC8_DIVISOR if crc_length == 8 else CRC16_DIVISOR
    if crc_length not in (8, 16):
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    r = crc_length
    padded = np.concatenate([info_bits, np.zeros(r, dtype=int)])
    return np.concatenate([info_bits, _crc_divide(padded, divisor)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    divisor = CRC8_DIVISOR if crc_length == 8 else CRC16_DIVISOR
    if crc_length not in (8, 16):
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return np.all(_crc_divide(bits.copy(), divisor) == 0)


def _g_tuple(L1, L2, b_tuple):
    """g 运算，b_tuple = (metric, [bits])"""
    bits = b_tuple[1]
    return [L2[i] + (1 - 2 * bits[i]) * L1[i] for i in range(len(L1))]


def _xor_paths(u1, u2, bits1, bits2):
    """合并左右子树路径"""
    res = [(u1[1][i] + u2[1][i]) % 2 for i in range(len(u1[1]))]
    res.extend(u2[1])
    return (u1[0] + u2[0], res, bits1 + bits2)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

        fb = np.asarray(frozen_bits)
        self.info_indices = np.where(fb == 0)[0] if fb.dtype != bool else np.where(~fb)[0]

        if crc_length > 0:
            self.crc_info_indices = self.info_indices[:len(self.info_indices) - crc_length]
        else:
            self.crc_info_indices = self.info_indices

    def _decode_scl(self, y, depth, node):
        """返回 (decisions, bit_lists)，decisions=[(pm, partial_bits), ...]"""
        if depth == self.n - 1:
            decisions = []
            bit_lists = []
            if node in self.frozen_set:
                decisions.append((0.0, [0]))
                bit_lists.append([0])
            else:
                if y[0] < 0:
                    decisions.append((0.0, [1]))
                    bit_lists.append([1])
                    decisions.append((abs(y[0]), [0]))
                    bit_lists.append([0])
                else:
                    decisions.append((0.0, [0]))
                    bit_lists.append([0])
                    decisions.append((abs(y[0]), [1]))
                    bit_lists.append([1])
            return decisions, bit_lists

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = f_operation(l1, l2)
        l_dec, l_bits = self._decode_scl(left, depth + 1, 2 * node)

        selection = []
        for i, l_d in enumerate(l_dec):
            right = _g_tuple(l1, l2, l_d)
            r_dec, r_bits = self._decode_scl(right, depth + 1, 2 * node + 1)
            for j, r_d in enumerate(r_dec):
                combined = _xor_paths(l_d, r_d, l_bits[i], r_bits[j])
                selection.append(combined)

        selection.sort(key=lambda x: x[0])
        selection = selection[:self.list_size]

        decisions = [(s[0], s[1]) for s in selection]
        bit_lists = [s[2] for s in selection]
        return decisions, bit_lists

    def decode(self, llr_ch):
        """返回 (u_hat, pm)"""
        if self.list_size == 1:
            fb = [1 if i in self.frozen_set else 0 for i in range(self.N)]
            return sc_decode(llr_ch, fb), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decisions, bit_lists = self._decode_scl(llr_ch, 0, 0)

        candidates = list(zip(decisions, bit_lists))

        if self.crc_length > 0:
            valid = []
            for (pm, _), bits in candidates:
                nv = np.array(bits, dtype=int)
                if crc_check(nv[self.crc_info_indices], self.crc_length):
                    valid.append((pm, nv))
            if valid:
                pm, nv = min(valid, key=lambda x: x[0])
            else:
                pm, nv = min([(d[0], np.array(b, int)) for d, b in candidates], key=lambda x: x[0])
        else:
            pm, nv = min([(d[0], np.array(b, int)) for d, b in candidates], key=lambda x: x[0])

        return nv, pm
