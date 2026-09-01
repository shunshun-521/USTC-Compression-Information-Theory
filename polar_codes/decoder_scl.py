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
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1

    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _g_with_decision(L1, L2, decision):
    _, bits = decision
    return list(g_operation(L1, L2, bits))


def _xor_decisions(left, right, left_nodes, right_nodes):
    _, bits_l = left
    _, bits_r = right
    merged = [((bits_l[i] + bits_r[i]) % 2) for i in range(len(bits_l))]
    merged.extend(bits_r)
    return (left[0] + right[0], merged, left_nodes + right_nodes)


def _scl_decode_recursive(y, depth, node, n, frozen_set, list_size):
    if depth == n - 1:
        decisions = []
        node_lists = []
        if node in frozen_set:
            if y[0] >= 0:
                decisions.append((0.0, [0]))
                node_lists.append([0])
            else:
                decisions.append((abs(y[0]), [0]))
                node_lists.append([0])
        else:
            if y[0] < 0:
                decisions.append((0.0, [1]))
                node_lists.append([1])
                decisions.append((abs(y[0]), [0]))
                node_lists.append([0])
            else:
                decisions.append((0.0, [0]))
                node_lists.append([0])
                decisions.append((abs(y[0]), [1]))
                node_lists.append([1])
        return decisions, node_lists

    half = len(y) // 2
    L1, L2 = y[:half], y[half:]
    left_llr = list(f_operation(L1, L2))

    l_decisions, l_node_lists = _scl_decode_recursive(
        left_llr, depth + 1, 2 * node, n, frozen_set, list_size
    )

    selection = []
    for i, decision in enumerate(l_decisions):
        right_llr = _g_with_decision(L1, L2, decision)
        r_decisions, r_node_lists = _scl_decode_recursive(
            right_llr, depth + 1, 2 * node + 1, n, frozen_set, list_size
        )
        for j, r_decision in enumerate(r_decisions):
            selection.append(
                _xor_decisions(decision, r_decision, l_node_lists[i], r_node_lists[j])
            )

    selection.sort(key=lambda x: x[0])
    selection = selection[:list_size]
    return (
        [(item[0], item[1]) for item in selection],
        [item[2] for item in selection],
    )


class SCLDecoder:
    """
    SCL 译码器（递归列表实现）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decisions, decoded_lists = _scl_decode_recursive(
            list(llr_ch),
            0,
            0,
            self.n,
            self.frozen_set,
            self.list_size,
        )

        if not decoded_lists:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = []
            for idx, bits in enumerate(decoded_lists):
                if len(bits) != self.N:
                    continue
                pm = decisions[idx][0] if idx < len(decisions) else 0.0
                payload = np.array(bits, dtype=int)[self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, bits))
            if valid:
                pm, bits = min(valid, key=lambda x: x[0])
            else:
                idx = 0
                pm = decisions[0][0] if decisions else 0.0
                bits = decoded_lists[0]
        else:
            idx = int(np.argmin([d[0] for d in decisions])) if decisions else 0
            pm = decisions[idx][0] if decisions else 0.0
            bits = decoded_lists[idx]

        u_hat = np.zeros(self.N, dtype=int)
        if len(bits) == self.N:
            u_hat[:] = bits
        return u_hat, pm
