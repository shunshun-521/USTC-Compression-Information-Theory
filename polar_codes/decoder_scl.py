"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _NodeState, f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    reg = 0
    for b in payload:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    expected = bits[-crc_length:]
    expected_val = sum(int(b) << (crc_length - 1 - i) for i, b in enumerate(expected))
    return reg == expected_val


class _SCLPath:
    __slots__ = ("pm", "beliefs", "decoded", "node_state", "node", "depth", "done")

    def __init__(self, n, N, llr_ch):
        self.pm = 0.0
        self.beliefs = np.zeros((n + 1, N), dtype=np.float64)
        self.decoded = np.zeros((n + 1, N), dtype=np.int32)
        self.node_state = np.zeros(2 * N - 1, dtype=np.int32)
        self.beliefs[0, :] = llr_ch
        self.node = 0
        self.depth = 0
        self.done = False

    def clone(self):
        p = _SCLPath.__new__(_SCLPath)
        p.pm = self.pm
        p.beliefs = self.beliefs.copy()
        p.decoded = self.decoded.copy()
        p.node_state = self.node_state.copy()
        p.node = self.node
        p.depth = self.depth
        p.done = self.done
        return p


def _advance_to_leaf(path, n, N, frozen_bits):
    """将路径推进到下一个叶节点，返回叶索引与叶 LLR"""
    while not path.done and path.depth < n:
        node = path.node
        depth = path.depth
        node_pos = (1 << depth) - 1 + node
        span = 1 << (n - depth)
        start = span * node
        end = start + span

        if path.node_state[node_pos] == _NodeState.NOT_VISITED:
            left = path.beliefs[depth, start : start + span // 2]
            right = path.beliefs[depth, start + span // 2 : end]
            child = node * 2
            child_depth = depth + 1
            child_span = span // 2
            child_start = child_span * child
            path.beliefs[child_depth, child_start : child_start + child_span] = f_operation(
                left, right
            )
            path.node_state[node_pos] = _NodeState.AFTER_L
            path.node = child
            path.depth += 1

        elif path.node_state[node_pos] == _NodeState.AFTER_L:
            left = path.beliefs[depth, start : start + span // 2]
            right = path.beliefs[depth, start + span // 2 : end]
            left_child = node * 2
            left_depth = depth + 1
            left_span = span // 2
            left_start = left_span * left_child
            left_bits = path.decoded[left_depth, left_start : left_start + left_span]

            child = node * 2 + 1
            child_depth = depth + 1
            child_span = span // 2
            child_start = child_span * child
            path.beliefs[child_depth, child_start : child_start + child_span] = g_operation(
                left, right, left_bits
            )
            path.node_state[node_pos] = _NodeState.AFTER_R
            path.node = child
            path.depth += 1

        else:
            left_child = node * 2
            right_child = node * 2 + 1
            parent_depth = depth + 1
            parent_span = span // 2
            left_start = parent_span * left_child
            right_start = parent_span * right_child
            left_bits = path.decoded[parent_depth, left_start : left_start + parent_span]
            right_bits = path.decoded[parent_depth, right_start : right_start + parent_span]
            path.decoded[depth, start:end] = np.concatenate(
                [(left_bits + right_bits) % 2, right_bits]
            )
            path.node //= 2
            path.depth -= 1

    leaf = path.node
    llr_leaf = path.beliefs[n, leaf]
    return leaf, llr_leaf


def _step_up_after_decision(path, n, N):
    """叶节点判决后向上回溯"""
    if path.node == N - 1:
        path.done = True
        return
    path.node //= 2
    path.depth -= 1
    while path.depth >= 0:
        node = path.node
        depth = path.depth
        node_pos = (1 << depth) - 1 + node
        if path.node_state[node_pos] == _NodeState.AFTER_R:
            span = 1 << (n - depth)
            start = span * node
            end = start + span
            left_child = node * 2
            right_child = node * 2 + 1
            parent_depth = depth + 1
            parent_span = span // 2
            left_start = parent_span * left_child
            right_start = parent_span * right_child
            left_bits = path.decoded[parent_depth, left_start : left_start + parent_span]
            right_bits = path.decoded[parent_depth, right_start : right_start + parent_span]
            path.decoded[depth, start:end] = np.concatenate(
                [(left_bits + right_bits) % 2, right_bits]
            )
            path.node //= 2
            path.depth -= 1
        else:
            break


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.n, self.N, llr_ch)]

        for _bit_round in range(self.N):
            candidates = []
            for path in paths:
                if path.done:
                    candidates.append(path)
                    continue

                leaf, llr_leaf = _advance_to_leaf(path, self.n, self.N, self.frozen_bits)
                bits = [0] if self.frozen_bits[leaf] else [0, 1]

                for bit in bits:
                    new_path = path.clone()
                    new_path.pm += self._pm_penalty(llr_leaf, bit)
                    new_path.decoded[self.n, leaf] = bit
                    _step_up_after_decision(new_path, self.n, self.N)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.decoded[self.n, :].astype(np.int32)

        if self.crc_length > 0:
            info_bits = u_hat[~self.frozen_bits]
            valid = [
                p
                for p in paths
                if crc_check(p.decoded[self.n, ~self.frozen_bits], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
                u_hat = best.decoded[self.n, :].astype(np.int32)

        return u_hat, best.pm
