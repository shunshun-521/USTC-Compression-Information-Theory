"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
from enum import IntEnum

import numpy as np

from decoder_sc import _TreeSCDecoder, f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0

    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(expected, bits)


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


def _clone_decoder(dec):
    """复制树 SC 译码器状态"""
    other = _TreeSCDecoder(dec.N, dec.frozen_bits)
    other.beliefs = dec.beliefs.copy()
    other.decoded_bits = dec.decoded_bits.copy()
    other.node_state = dec.node_state.copy()
    other.u_hat = dec.u_hat.copy()
    if hasattr(dec, "cursor"):
        other.cursor = dec.cursor
    return other


def _advance_to_leaf(dec, target_node):
    """从当前游标推进到指定叶节点并返回 LLR"""
    if hasattr(dec, "cursor"):
        node, depth = dec.cursor
    else:
        node, depth = 0, 0

    n = dec.n
    while True:
        if depth == n and node == target_node:
            dec.cursor = (node, depth)
            return dec.beliefs[n, node]

        if depth == n:
            node //= 2
            depth -= 1
            continue

        node_pos = 2 ** depth - 1 + node
        state = dec.node_state[node_pos]
        if state == _NodeState.NOT_VISITED:
            dec._step_l(node, depth, node_pos)
            node *= 2
            depth += 1
        elif state == _NodeState.AFTER_L:
            dec._step_r(node, depth, node_pos)
            node = node * 2 + 1
            depth += 1
        else:
            dec._step_u(node, depth)
            node //= 2
            depth -= 1


def _decide_leaf(dec, phi, bit):
    """在叶节点做出判决并沿树回退一步（与 SC 一致）"""
    dec.u_hat[phi] = bit
    dec.decoded_bits[dec.n, phi] = bit
    node, depth = dec.cursor
    if phi < dec.N - 1:
        node //= 2
        depth -= 1
    dec.cursor = (node, depth)


class SCLDecoder:
    """SCL 译码器（基于树 SC 增量推进）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{"dec": _TreeSCDecoder(self.N, self.frozen_bits), "pm": 0.0}]
        paths[0]["dec"].beliefs[0, :] = llr_ch

        for phi in range(self.N):
            extended = []
            for path in paths:
                dec = path["dec"]
                llr = _advance_to_leaf(dec, phi)

                if self.frozen_bits[phi]:
                    bit = 0
                    _decide_leaf(dec, phi, bit)
                    extended.append(
                        {"dec": dec, "pm": path["pm"] + self._pm_penalty(llr, bit)}
                    )
                else:
                    for bit in (0, 1):
                        new_dec = _clone_decoder(dec)
                        _decide_leaf(new_dec, phi, bit)
                        extended.append(
                            {
                                "dec": new_dec,
                                "pm": path["pm"] + self._pm_penalty(llr, bit),
                            }
                        )

            extended.sort(key=lambda p: p["pm"])
            paths = extended[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u = path["dec"].u_hat.copy()
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                return best["dec"].u_hat.copy(), best["pm"]

        best = min(paths, key=lambda p: p["pm"])
        return best["dec"].u_hat.copy(), best["pm"]
