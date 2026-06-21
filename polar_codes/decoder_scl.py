"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import NodeState, f_operation, g_operation, sc_decode


CRC_POLYNOMIALS = {
    8: [1, 0, 0, 0, 0, 0, 1, 1, 1],
    16: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def _gf2_remainder(msg, poly):
    msg = list(map(int, msg))
    poly = list(map(int, poly))
    n = len(poly) - 1
    while True:
        try:
            i = msg.index(1)
        except ValueError:
            return msg[-n:]
        if len(msg) - i < len(poly):
            break
        for j in range(len(poly)):
            msg[i + j] ^= poly[j]
    return msg[-n:]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _gf2_remainder(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly)
    return np.concatenate([info_bits, np.asarray(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return sum(_gf2_remainder(bits, CRC_POLYNOMIALS[crc_length])) == 0


class Path:
    """SCL 路径状态。"""

    __slots__ = ("beliefs", "decoded_bits", "node_state", "pm", "u_hat", "node", "depth", "done")

    def __init__(self, N, n, llr_ch):
        self.beliefs = np.zeros((n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((n + 1, N), dtype=np.int32)
        self.node_state = np.zeros(2 * N - 1, dtype=np.int32)
        self.beliefs[0, :] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.node = 0
        self.depth = 0
        self.done = False

    def copy(self):
        p = Path(self.beliefs.shape[1], int(math.log2(self.beliefs.shape[1])), self.beliefs[0])
        p.beliefs = self.beliefs.copy()
        p.decoded_bits = self.decoded_bits.copy()
        p.node_state = self.node_state.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        p.node = self.node
        p.depth = self.depth
        p.done = self.done
        return p


def _sc_step(path, n, N, frozen_bits):
    """执行 SC 树遍历的一步，返回 'leaf' / 'continue' / 'done'。"""
    if path.done:
        return "done"

    if path.depth == n:
        phi = path.node
        llr_phi = path.beliefs[n, phi]
        if frozen_bits[phi]:
            path.pm += 0.0 if llr_phi >= 0 else abs(llr_phi)
            path.u_hat[phi] = 0
            path.decoded_bits[n, phi] = 0
        else:
            return "leaf", phi, llr_phi

        if phi == N - 1:
            path.done = True
            return "done"
        path.node //= 2
        path.depth -= 1
        return "continue"

    node_pos = (1 << path.depth) - 1 + path.node
    if path.node_state[node_pos] == NodeState.NOT_VISITED:
        span = 1 << (n - path.depth)
        incoming = path.beliefs[path.depth, span * path.node:span * (path.node + 1)]
        half = span // 2
        child = path.node * 2
        child_span = span // 2
        path.beliefs[path.depth + 1, child_span * child:child_span * (child + 1)] = f_operation(
            incoming[:half], incoming[half:]
        )
        path.node_state[node_pos] = NodeState.AFTER_L
        path.node = child
        path.depth += 1
    elif path.node_state[node_pos] == NodeState.AFTER_L:
        span = 1 << (n - path.depth)
        incoming = path.beliefs[path.depth, span * path.node:span * (path.node + 1)]
        half = span // 2
        left_child = path.node * 2
        left_span = span // 2
        u_left = path.decoded_bits[path.depth + 1, left_span * left_child:left_span * (left_child + 1)]
        child = path.node * 2 + 1
        child_span = span // 2
        path.beliefs[path.depth + 1, child_span * child:child_span * (child + 1)] = g_operation(
            incoming[:half], incoming[half:], u_left
        )
        path.node_state[node_pos] = NodeState.AFTER_R
        path.node = child
        path.depth += 1
    else:
        span = 1 << (n - path.depth)
        left_child = path.node * 2
        right_child = path.node * 2 + 1
        half = span // 2
        bits_left = path.decoded_bits[path.depth + 1, half * left_child:half * (left_child + 1)]
        bits_right = path.decoded_bits[path.depth + 1, half * right_child:half * (right_child + 1)]
        path.decoded_bits[path.depth, span * path.node:span * (path.node + 1)] = np.concatenate(
            [(bits_left + bits_right) % 2, bits_right]
        )
        path.node //= 2
        path.depth -= 1

    return "continue"


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _branch_penalty(llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        while not all(p.done for p in paths):
            new_paths = []
            leaf_events = []

            for path in paths:
                if path.done:
                    new_paths.append(path)
                    continue

                result = _sc_step(path, self.n, self.N, self.frozen_bits)
                if isinstance(result, tuple) and result[0] == "leaf":
                    _, phi, llr_phi = result
                    leaf_events.append((path, phi, llr_phi))
                elif result == "done":
                    new_paths.append(path)
                else:
                    new_paths.append(path)

            if leaf_events:
                expanded = []
                for path, phi, llr_phi in leaf_events:
                    if self.frozen_bits[phi]:
                        expanded.append(path)
                    else:
                        for u in (0, 1):
                            p2 = path.copy()
                            p2.pm += self._branch_penalty(llr_phi, u)
                            p2.u_hat[phi] = u
                            p2.decoded_bits[self.n, phi] = u
                            if phi == self.N - 1:
                                p2.done = True
                            else:
                                p2.node //= 2
                                p2.depth = self.n - 1
                            expanded.append(p2)
                expanded.sort(key=lambda p: p.pm)
                paths = expanded[: self.list_size]
            else:
                paths = new_paths

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p.u_hat[info_pos], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
