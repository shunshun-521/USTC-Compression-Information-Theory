"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _NodeState,
    f_operation,
    g_operation,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _PathState:
    __slots__ = ('pm', 'beliefs', 'decoded_bits', 'node_state')

    def __init__(self, n, N):
        self.pm = 0.0
        self.beliefs = np.zeros((n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((n + 1, N), dtype=np.int8)
        self.node_state = np.zeros(2 * N - 1, dtype=int)


def _copy_path(path):
    new_path = _PathState(path.beliefs.shape[0] - 1, path.beliefs.shape[1])
    new_path.pm = path.pm
    new_path.beliefs = path.beliefs.copy()
    new_path.decoded_bits = path.decoded_bits.copy()
    new_path.node_state = path.node_state.copy()
    return new_path


def _metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _step_l(path, n, node, depth, node_pos):
    span = 1 << (n - depth)
    incoming = path.beliefs[depth, span * node:span * (node + 1)]
    half = span // 2
    left_child = 2 * node
    child_depth = depth + 1
    child_span = span // 2
    path.beliefs[child_depth, child_span * left_child:child_span * (left_child + 1)] = f_operation(
        incoming[:half], incoming[half:]
    )
    path.node_state[node_pos] = _NodeState.AFTER_L


def _step_r(path, n, node, depth, node_pos):
    span = 1 << (n - depth)
    incoming = path.beliefs[depth, span * node:span * (node + 1)]
    half = span // 2
    left_child = 2 * node
    child_depth = depth + 1
    child_span = span // 2
    decoded_left = path.decoded_bits[child_depth, child_span * left_child:child_span * (left_child + 1)]
    right_child = 2 * node + 1
    path.beliefs[child_depth, child_span * right_child:child_span * (right_child + 1)] = g_operation(
        incoming[:half], incoming[half:], decoded_left
    )
    path.node_state[node_pos] = _NodeState.AFTER_R


def _step_u(path, n, node, depth):
    span = 1 << (n - depth)
    child_depth = depth + 1
    child_span = span // 2
    left_child = 2 * node
    right_child = 2 * node + 1
    left_bits = path.decoded_bits[child_depth, child_span * left_child:child_span * (left_child + 1)]
    right_bits = path.decoded_bits[child_depth, child_span * right_child:child_span * (right_child + 1)]
    path.decoded_bits[depth, span * node:span * (node + 1)] = np.concatenate(
        [(left_bits + right_bits) % 2, right_bits]
    )


class SCLDecoder:
    """SCL 译码器（树遍历，多路径）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.n, self.N)]
        paths[0].beliefs[0, :] = llr_ch

        node = 0
        depth = 0
        done = False

        while not done:
            if depth == self.n:
                new_paths = []
                for path in paths:
                    llr = path.beliefs[self.n, node]
                    if self.frozen_bits[node]:
                        p = _copy_path(path)
                        p.pm += _metric_penalty(llr, 0)
                        p.decoded_bits[self.n, node] = 0
                        new_paths.append(p)
                    else:
                        for bit in (0, 1):
                            p = _copy_path(path)
                            p.pm += _metric_penalty(llr, bit)
                            p.decoded_bits[self.n, node] = bit
                            new_paths.append(p)
                paths = sorted(new_paths, key=lambda p: p.pm)[: self.list_size]

                if node == self.N - 1:
                    done = True
                else:
                    node //= 2
                    depth -= 1
            else:
                node_pos = (1 << depth) - 1 + node
                expanded = []
                for path in paths:
                    state = path.node_state[node_pos]
                    p = _copy_path(path)
                    if state == _NodeState.NOT_VISITED:
                        _step_l(p, self.n, node, depth, node_pos)
                        expanded.append(p)
                    elif state == _NodeState.AFTER_L:
                        _step_r(p, self.n, node, depth, node_pos)
                        expanded.append(p)
                    else:
                        _step_u(p, self.n, node, depth)
                        expanded.append(p)
                paths = expanded

                state = paths[0].node_state[node_pos]
                if state == _NodeState.AFTER_L:
                    node = node * 2 + 1
                    depth += 1
                elif state == _NodeState.NOT_VISITED:
                    node *= 2
                    depth += 1
                else:
                    node //= 2
                    depth -= 1

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.decoded_bits[self.n, self.info_positions], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.decoded_bits[self.n, :].astype(int), best.pm
