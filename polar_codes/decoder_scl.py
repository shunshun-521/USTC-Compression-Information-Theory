"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _get_tree,
    _set_frozen,
    f_operation,
    g_operation,
    sc_decode,
)


def _crc16_poly_bits():
    p = [0] * 17
    p[0] = p[1] = p[14] = p[16] = 1
    return p


def _get_poly(crc_length):
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 0, 1, 1, 1]
    if crc_length == 16:
        return _crc16_poly_bits()
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（GF(2) 长除）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _get_poly(crc_length)
    msg = list(map(int, info_bits)) + [0] * crc_length
    n_info = len(info_bits)
    for i in range(n_info):
        if msg[i]:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    crc_bits = np.array(msg[n_info:], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _get_poly(crc_length)
    msg = list(map(int, bits))
    n = len(msg) - crc_length
    for i in range(n):
        if msg[i]:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return sum(msg[n:]) == 0


def _decode_subtree_with_prefix(node, u_prefix, frozen_bits):
    """译码子树全部叶节点（前缀比特来自 u_prefix，否则硬判决）。"""
    if node.size == 1:
        lid = node.lane_id
        if frozen_bits[lid]:
            node.s[0] = 0
        elif lid in u_prefix:
            node.s[0] = u_prefix[lid]
        else:
            node.s[0] = 0 if node.lambda_[0] >= 0 else 1
        return

    half = node.size // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[i + half])
    _decode_subtree_with_prefix(node.left, u_prefix, frozen_bits)
    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[i + half], node.left.s[i]
        )
    _decode_subtree_with_prefix(node.right, u_prefix, frozen_bits)
    for i in range(half):
        node.s[i] = (node.left.s[i] ^ node.right.s[i]) % 2
        node.s[i + half] = node.right.s[i]


def _llr_at_lane(node, target_lane, u_prefix, frozen_bits):
    """在已判决前缀 u_prefix 下，计算 target_lane 的 LLR。"""
    if node.size == 1:
        lid = node.lane_id
        if lid < target_lane:
            if frozen_bits[lid]:
                node.s[0] = 0
            else:
                node.s[0] = u_prefix[lid]
            return None
        if lid == target_lane:
            return float(node.lambda_[0])
        return None

    half = node.size // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[i + half])

    if _lane_in_subtree(node.left, target_lane):
        val = _llr_at_lane(node.left, target_lane, u_prefix, frozen_bits)
        if val is not None:
            return val
    else:
        _decode_subtree_with_prefix(node.left, u_prefix, frozen_bits)

    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[i + half], node.left.s[i]
        )
    return _llr_at_lane(node.right, target_lane, u_prefix, frozen_bits)


def _min_lane(node):
    if node.size == 1:
        return node.lane_id
    return _min_lane(node.left)


def _max_lane(node):
    if node.size == 1:
        return node.lane_id
    return _max_lane(node.right)


def _lane_in_subtree(node, lane):
    return _min_lane(node) <= lane <= _max_lane(node)


def _llr_for_bit(llr_ch, u_prefix, lane, frozen_bits, N):
    root = _get_tree(N)
    _set_frozen(root, frozen_bits)
    root.lambda_[:] = llr_ch
    return _llr_at_lane(root, lane, u_prefix, frozen_bits)


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "u": {}}]

        for lane in range(self.N):
            candidates = []
            for path in paths:
                llr_bit = _llr_for_bit(llr_ch, path["u"], lane, self.frozen_bits, self.N)
                if self.frozen_bits[lane]:
                    pen = abs(llr_bit) if llr_bit < 0 else 0.0
                    new_u = dict(path["u"])
                    new_u[lane] = 0
                    candidates.append({"pm": path["pm"] + pen, "u": new_u})
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit)
                        new_u = dict(path["u"])
                        new_u[lane] = bit
                        candidates.append({"pm": path["pm"] + pen, "u": new_u})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.L]

        u_hat = np.zeros(self.N, dtype=int)
        for lane in range(self.N):
            if self.frozen_bits[lane]:
                u_hat[lane] = 0
            else:
                u_hat[lane] = paths[0]["u"].get(lane, 0)

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for path in paths:
                for lane in range(self.N):
                    u_hat[lane] = 0 if self.frozen_bits[lane] else path["u"].get(lane, 0)
                bits = u_hat[info_mask]
                if crc_check(bits, self.crc_length):
                    return u_hat, path["pm"]

        best = paths[0]
        for lane in range(self.N):
            u_hat[lane] = 0 if self.frozen_bits[lane] else best["u"].get(lane, 0)
        return u_hat, best["pm"]
