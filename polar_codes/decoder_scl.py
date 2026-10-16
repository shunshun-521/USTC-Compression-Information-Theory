"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
import math

from decoder_sc import (
    f_operation,
    g_operation,
    _SCNode,
    _build_tree,
    _recursive_decode_aff3ct,
    _collect_leaves,
    _hard_bit,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


def _clone_tree(node):
    new = _SCNode(node.size, node.lane_id, node.is_frozen)
    new.lambda_v = node.lambda_v.copy()
    new.s = node.s.copy()
    if not node.leaf:
        new.left = _clone_tree(node.left)
        new.right = _clone_tree(node.right)
    return new


def _find_leaf(root, phi):
    if root.leaf:
        return root if root.lane_id == phi else None
    return _find_leaf(root.left, phi) or _find_leaf(root.right, phi)


def _sc_step_to_phi(root, llr_ch, phi):
    """将树推进到即将判决第 phi 位（含先前已判决比特）"""

    def visit(node):
        if node.leaf:
            if node.lane_id < phi:
                return
            if node.lane_id == phi:
                return
            if node.is_frozen:
                node.s[0] = 0
            else:
                node.s[0] = _hard_bit(node.lambda_v[0])
            return

        half = node.size // 2
        for i in range(half):
            node.left.lambda_v[i] = f_operation(
                node.lambda_v[i], node.lambda_v[half + i]
            )
        visit(node.left)
        for i in range(half):
            node.right.lambda_v[i] = g_operation(
                node.lambda_v[i], node.lambda_v[half + i], node.left.s[i]
            )
        visit(node.right)
        for i in range(half):
            node.s[i] = (node.left.s[i] ^ node.right.s[i]) & 1
            node.s[half + i] = node.right.s[i]

    root.lambda_v[:] = llr_ch
    visit(root)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, list_size)
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        lane_counter = [0]
        root_template = _build_tree(self.n, self.frozen_bits, lane_counter)

        paths = [{"pm": 0.0, "root": _clone_tree(root_template)}]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                root = path["root"]
                _sc_step_to_phi(root, llr_ch, phi)
                leaf = _find_leaf(root, phi)
                llr_phi = leaf.lambda_v[0]

                if self.frozen_bits[phi]:
                    pm = path["pm"] + (abs(llr_phi) if llr_phi < 0 else 0.0)
                    leaf.s[0] = 0
                    expanded.append({"pm": pm, "root": root})
                else:
                    for bit in (0, 1):
                        nr = _clone_tree(root)
                        lf = _find_leaf(nr, phi)
                        lf.s[0] = bit
                        pen = 0.0 if (bit == 0 and llr_phi >= 0) or (bit == 1 and llr_phi < 0) else abs(llr_phi)
                        expanded.append({"pm": path["pm"] + pen, "root": nr})

            expanded.sort(key=lambda x: x["pm"])
            paths = expanded[: self.L]

        results = []
        for path in paths:
            _recursive_decode_aff3ct(path["root"])
            u = np.zeros(self.N, dtype=int)
            _collect_leaves(path["root"], u)
            results.append((path["pm"], u))

        if self.crc_length > 0:
            ok = [
                (pm, u)
                for pm, u in results
                if crc_check(u[self.info_positions], self.crc_length)
            ]
            if ok:
                ok.sort(key=lambda x: x[0])
                return ok[0][1], ok[0][0]

        results.sort(key=lambda x: x[0])
        return results[0][1], results[0][0]
