"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _g_path(L1, L2, decision):
    """g 运算，decision = (pm, bits)。"""
    pm, bits = decision
    return g_operation(L1, L2, bits)


def _xor_paths(u1, u2, list1, list2):
    """合并左右子树路径。"""
    pm = u1[0] + u2[0]
    bits = [(u1[1][i] + u2[1][i]) % 2 for i in range(len(u1[1]))]
    bits.extend(u2[1])
    return (pm, bits, list1 + list2)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _decode_tree(self, y, depth, node):
        if depth == self.n - 1:
            decisions = []
            decoded_lists = []
            if node in self.frozen_set:
                if y[0] >= 0:
                    decisions.append((0.0, [0]))
                else:
                    decisions.append((abs(y[0]), [0]))
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

        y = np.asarray(y, dtype=np.float64)
        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)

        left_decisions, left_lists = self._decode_tree(left_llr, depth + 1, 2 * node)

        selection = []
        for i, left_dec in enumerate(left_decisions):
            right_llr = _g_path(L1, L2, left_dec)
            right_decisions, right_lists = self._decode_tree(
                right_llr, depth + 1, 2 * node + 1
            )
            for j, right_dec in enumerate(right_decisions):
                selection.append(
                    _xor_paths(left_dec, right_dec, left_lists[i], right_lists[j])
                )

        selection.sort(key=lambda x: x[0])
        if len(selection) > self.list_size:
            selection = selection[: self.list_size]

        decisions = [(s[0], s[1]) for s in selection]
        decoded_lists = [s[2] for s in selection]
        return decisions, decoded_lists

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decisions, decoded_lists = self._decode_tree(llr_ch, 0, 0)

        if not decoded_lists:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            valid_idx = [
                i for i, dl in enumerate(decoded_lists)
                if crc_check(np.array(dl)[self.info_indices], self.crc_length)
            ]
            if valid_idx:
                best_i = min(valid_idx, key=lambda i: decisions[i][0])
            else:
                best_i = min(range(len(decisions)), key=lambda i: decisions[i][0])
        else:
            best_i = min(range(len(decisions)), key=lambda i: decisions[i][0])

        u_hat = np.zeros(self.N, dtype=int)
        codeword = decoded_lists[best_i]
        if len(codeword) == self.N:
            u_hat = np.array(codeword, dtype=int)
        else:
            for i, bit in enumerate(codeword):
                if i < self.N:
                    u_hat[i] = bit

        return u_hat, decisions[best_i][0]
