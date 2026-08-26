"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    h_operation,
    _TreeNode,
    _build_tree,
    _init_frozen,
    sc_decode,
    precompute_sc_indices,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _max_lane(node):
    if node.is_leaf:
        return node.lane_id
    return max(_max_lane(node.left), _max_lane(node.right))


def _full_decode_subtree(node, u_hat, frozen_bits):
    """完整译码子树（使用已知 u_hat）。"""
    if node.is_leaf:
        node.s[0] = 0 if frozen_bits[node.lane_id] else int(u_hat[node.lane_id])
        return

    half = len(node.lambda_v) // 2
    node.left.lambda_v[:] = f_operation(node.lambda_v[:half], node.lambda_v[half:])
    _full_decode_subtree(node.left, u_hat, frozen_bits)
    node.right.lambda_v[:] = g_operation(
        node.lambda_v[:half], node.lambda_v[half:], node.left.s
    )
    _full_decode_subtree(node.right, u_hat, frozen_bits)
    node.s[:half] = node.left.s ^ node.right.s
    node.s[half:] = node.right.s


def _llr_at_phi(node, phi, u_hat, frozen_bits):
    """获取第 phi 位 LLR（已知 u_hat[0:phi]）。"""
    if node.is_leaf:
        if node.lane_id == phi:
            return float(node.lambda_v[0])
        return None

    half = len(node.lambda_v) // 2
    node.left.lambda_v[:] = f_operation(node.lambda_v[:half], node.lambda_v[half:])

    if phi <= _max_lane(node.left):
        return _llr_at_phi(node.left, phi, u_hat, frozen_bits)

    _full_decode_subtree(node.left, u_hat, frozen_bits)
    node.right.lambda_v[:] = g_operation(
        node.lambda_v[:half], node.lambda_v[half:], node.left.s
    )
    return _llr_at_phi(node.right, phi, u_hat, frozen_bits)


class SCLDecoder:
    """SCL 译码器（逐位 LLR + 路径列表裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self._frozen_bool = self.frozen_bits.astype(bool)

    def _make_root(self, llr_ch):
        root = _build_tree(self.n, [0])
        _init_frozen(root, self._frozen_bool)
        root.lambda_v[:] = llr_ch
        return root

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for phi in range(self.N):
            candidates = []
            for pm, u_hat in paths:
                root = self._make_root(llr_ch)
                llr = _llr_at_phi(root, phi, u_hat, self._frozen_bool)
                if self.frozen_bits[phi]:
                    new_u = u_hat.copy()
                    new_u[phi] = 0
                    new_pm = pm + (0.0 if llr >= 0 else abs(llr))
                    candidates.append((new_pm, new_u))
                else:
                    for u_val in (0, 1):
                        hard = h_operation(llr)
                        new_pm = pm if u_val == hard else pm + abs(llr)
                        new_u = u_hat.copy()
                        new_u[phi] = u_val
                        candidates.append((new_pm, new_u))
            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
            else:
                pm, u_hat = min(paths, key=lambda x: x[0])
        else:
            pm, u_hat = min(paths, key=lambda x: x[0])

        return u_hat.astype(int), pm
