"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import copy
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    h_operation,
    _build_tree,
    _collect_leaves,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=int)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(bits, dtype=int)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _recursive_decode_partial(node, u_fixed, phi_limit):
    """译码至相位 phi_limit（含），此前相位使用 u_fixed 固定位"""
    if node.size > 1:
        half = node.size // 2
        left, right = node.left, node.right
        for i in range(half):
            left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[i + half])
        _recursive_decode_partial(left, u_fixed, phi_limit)
        for i in range(half):
            right.lambda_[i] = g_operation(
                node.lambda_[i], node.lambda_[i + half], left.s[i]
            )
        _recursive_decode_partial(right, u_fixed, phi_limit)
        for i in range(half):
            node.s[i] = left.s[i] ^ right.s[i]
            node.s[i + half] = right.s[i]
    else:
        lane = node.lane_id
        if lane < phi_limit:
            node.s[0] = int(u_fixed[lane])
        elif node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = int(h_operation(node.lambda_[0]))


def llr_at_phase(llr_ch, frozen_bits, u_prefix, phi):
    """给定前缀 u_prefix[0:phi]，返回相位 phi 的 LLR（树顶 f/g 传播后叶节点 lambda）"""
    N = len(llr_ch)
    root = _build_tree(N, frozen_bits)
    root.lambda_[:] = llr_ch
    _recursive_decode_partial(root, u_prefix, phi)
    # 找到 lane_id==phi 的叶节点 lambda
    def find_llr(node):
        if node.size == 1:
            return node.lambda_[0] if node.lane_id == phi else None
        v = find_llr(node.left)
        return v if v is not None else find_llr(node.right)

    return float(find_llr(root))


def _pm_update(llr_bit, bit):
    """路径度量：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr_bit >= 0 else 1
    return 0.0 if bit == hard else abs(llr_bit)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [(0.0, np.zeros(self.N, dtype=int))]
        for phi in range(self.N):
            candidates = []
            for pm, u in paths:
                llr_phi = llr_at_phase(llr_ch, self.frozen_bits, u, phi)
                if self.frozen_bits[phi]:
                    u_new = u.copy()
                    u_new[phi] = 0
                    candidates.append((pm + _pm_update(llr_phi, 0), u_new))
                else:
                    for bit in (0, 1):
                        u_new = u.copy()
                        u_new[phi] = bit
                        candidates.append((pm + _pm_update(llr_phi, bit), u_new))
            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in paths
                if crc_check(u[self.info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        best_pm, best_u = min(paths, key=lambda x: x[0])
        return best_u, best_pm
