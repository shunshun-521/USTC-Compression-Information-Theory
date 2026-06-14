"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _NodeState, f_operation, g_operation, sc_decode
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded[-crc_length:], bits[-crc_length:])


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class _TreeRunner:
    """增量式 SC 树遍历，供 SCL 按叶节点推进。"""

    def __init__(self, channel_br, frozen_bits):
        self.N = len(channel_br)
        self.n = int(math.log2(self.N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.beliefs = np.zeros((self.n + 1, self.N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, self.N), dtype=int)
        self.node_state = np.zeros(2 * self.N - 1, dtype=int)
        self.beliefs[0, :] = channel_br
        self.node = 0
        self.depth = 0
        self.next_phi = 0

    def copy(self):
        r = _TreeRunner(self.beliefs[0, :].copy(), self.frozen_bits)
        r.beliefs = self.beliefs.copy()
        r.decoded_bits = self.decoded_bits.copy()
        r.node_state = self.node_state.copy()
        r.node = self.node
        r.depth = self.depth
        r.next_phi = self.next_phi
        return r

    def _step_l(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left = 2 * node
        self.beliefs[depth + 1, (span // 2) * left : (span // 2) * (left + 1)] = (
            f_operation(incoming[:half], incoming[half:])
        )
        self.node_state[node_pos] = _NodeState.AFTER_L

    def _step_r(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left = 2 * node
        left_span = span // 2
        decoded_left = self.decoded_bits[
            depth + 1, left_span * left : left_span * (left + 1)
        ]
        right = 2 * node + 1
        self.beliefs[depth + 1, left_span * right : left_span * (right + 1)] = (
            g_operation(incoming[:half], incoming[half:], decoded_left)
        )
        self.node_state[node_pos] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 1 << (self.n - depth)
        half = span // 2
        left = 2 * node
        right = 2 * node + 1
        bits_left = self.decoded_bits[depth + 1, half * left : half * (left + 1)]
        bits_right = self.decoded_bits[depth + 1, half * right : half * (right + 1)]
        self.decoded_bits[depth, span * node : span * (node + 1)] = np.concatenate(
            [(bits_left + bits_right) % 2, bits_right]
        )

    def _tick(self, u_hat):
        if self.depth == self.n:
            if self.node < self.next_phi:
                self.decoded_bits[self.n, self.node] = u_hat[self.node]
            if self.node >= self.N - 1:
                return "done"
            self.node //= 2
            self.depth -= 1
            return "up"

        node_pos = (1 << self.depth) - 1 + self.node
        state = self.node_state[node_pos]
        if state == _NodeState.NOT_VISITED:
            self._step_l(self.node, self.depth, node_pos)
            self.node *= 2
            self.depth += 1
        elif state == _NodeState.AFTER_L:
            self._step_r(self.node, self.depth, node_pos)
            self.node = self.node * 2 + 1
            self.depth += 1
        else:
            self._step_u(self.node, self.depth)
            self.node //= 2
            self.depth -= 1
        return "continue"

    def llr_at_phi(self, u_hat, phi):
        assert phi == self.next_phi
        while not (self.depth == self.n and self.node == phi):
            self._tick(u_hat)
        return float(self.beliefs[self.n, phi])

    def commit(self, phi, bit):
        assert phi == self.next_phi
        self.decoded_bits[self.n, phi] = bit
        self.next_phi = phi + 1
        if phi == self.N - 1:
            return
        self.node //= 2
        self.depth -= 1


class SCLDecoder:
    """SCL 译码器（增量树遍历）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        channel_br = llr_ch[self.br]
        paths = [
            {
                "pm": 0.0,
                "u_hat": np.zeros(self.N, dtype=int),
                "runner": _TreeRunner(channel_br, self.frozen_bits),
            }
        ]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr = path["runner"].llr_at_phi(path["u_hat"], phi)
                if self.frozen_bits[phi]:
                    child = {
                        "pm": path["pm"] + _pm_penalty(llr, 0),
                        "u_hat": path["u_hat"].copy(),
                        "runner": path["runner"].copy(),
                    }
                    child["u_hat"][phi] = 0
                    child["runner"].commit(phi, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + _pm_penalty(llr, bit),
                            "u_hat": path["u_hat"].copy(),
                            "runner": path["runner"].copy(),
                        }
                        child["u_hat"][phi] = bit
                        child["runner"].commit(phi, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
