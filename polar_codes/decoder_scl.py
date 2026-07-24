"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (fb * poly)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _g_vec(L1, L2, decision):
    """g 运算，decision = (pm, bit_list)。"""
    bits = decision[1]
    return [L2[i] + (1 - 2 * bits[i]) * L1[i] for i in range(len(L2))]


def _xor_paths(u1, u2, u1_list, u2_list):
    res = [(u1[1][i] + u2[1][i]) % 2 for i in range(len(u1[1]))]
    res.extend(u2[1])
    return (u1[0] + u2[0], res, u1_list + u2_list)


class SCLDecoder:
    """SCL 译码器（HETSN 递归列表译码）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _decode(self, y, depth, node):
        if depth == self.n - 1:
            decisions = []
            decoded_lists = []
            if node in self.frozen_set:
                pm = 0.0 if y[0] >= 0 else abs(y[0])
                decisions.append((pm, [0]))
                decoded_lists.append([0])
            else:
                if y[0] < 0:
                    decisions.append((0.0, [1]))
                    decoded_lists.append([1])
                    decisions.append((abs(y[0]), [0]))
                    decoded_lists.append([0])
                else:
                    decisions.append((0.0, [0]))
                    decoded_lists.append([0])
                    decisions.append((abs(y[0]), [1]))
                    decoded_lists.append([1])
            return decisions, decoded_lists

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left_llr = f_operation(np.array(l1), np.array(l2)).tolist()
        l_dec, l_bits = self._decode(left_llr, depth + 1, 2 * node)

        selection = []
        for i, ld in enumerate(l_dec):
            right_llr = _g_vec(l1, l2, ld)
            r_dec, r_bits = self._decode(right_llr, depth + 1, 2 * node + 1)
            for j, rd in enumerate(r_dec):
                selection.append(_xor_paths(ld, rd, l_bits[i], r_bits[j]))

        selection.sort(key=lambda x: x[0])
        selection = selection[: self.list_size]
        return [(s[0], s[1]) for s in selection], [s[2] for s in selection]

    def decode(self, llr_ch):
        if self.list_size == 1:
            frozen_bits = np.zeros(self.N, dtype=bool)
            frozen_bits[list(self.frozen_set)] = True
            u_hat = sc_decode_recursive(llr_ch, frozen_bits)
            return u_hat, 0.0

        decisions, decoded_lists = self._decode(llr_ch.tolist(), 0, 0)
        if not decoded_lists:
            return np.zeros(self.N, dtype=int), 0.0

        best_idx = 0
        if self.crc_length > 0:
            valid = [
                i for i, bits in enumerate(decoded_lists)
                if len(bits) >= len(self.info_indices)
                and crc_check(np.array(bits)[: len(self.info_indices)], self.crc_length)
            ]
            if valid:
                best_idx = min(valid, key=lambda i: decisions[i][0])
            else:
                best_idx = int(np.argmin([d[0] for d in decisions]))
        else:
            best_idx = int(np.argmin([d[0] for d in decisions]))

        bits = decoded_lists[best_idx]
        u_hat = np.zeros(self.N, dtype=int)
        n_copy = min(len(bits), self.N)
        u_hat[:n_copy] = bits[:n_copy]
        return u_hat, decisions[best_idx][0]
