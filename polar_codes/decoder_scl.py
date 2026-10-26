"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    permute_llr_for_decode,
    sc_decode_recursive,
    _get_tree,
    _init_frozen_flags,
    _h_hard,
)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.uint8)
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
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _llr_at_phase(llr, frozen_bits, u_prefix, phi):
    """已知 u[0:phi] 时，返回比特 phi 的 LLR"""
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr)
    root = _get_tree(N)
    _init_frozen_flags(root, frozen_bits)
    root.lambda_ = np.array(llr, dtype=np.float64)
    root.s = np.zeros(N, dtype=np.int8)

    def decode_until(node, offset):
        if node.is_leaf():
            idx = offset
            if idx < phi:
                node.s[0] = u_prefix[idx]
            else:
                return node.lambda_[0]
            return None
        h = node.size // 2
        if node.left.lambda_ is None:
            node.left.lambda_ = np.zeros(h, dtype=np.float64)
            node.left.s = np.zeros(h, dtype=np.int8)
            node.right.lambda_ = np.zeros(h, dtype=np.float64)
            node.right.s = np.zeros(h, dtype=np.int8)
        for i in range(h):
            node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[h + i])
        res = decode_until(node.left, offset)
        if res is not None:
            return res
        for i in range(h):
            node.right.lambda_[i] = g_operation(
                node.lambda_[i], node.lambda_[h + i], node.left.s[i]
            )
        res = decode_until(node.right, offset + h)
        if res is not None:
            return res
        for i in range(h):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[h + i] = node.right.s[i]
        return None

    out = decode_until(root, 0)
    return float(out)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.L = max(1, list_size)
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr = permute_llr_for_decode(llr_ch)
        frozen = self.frozen_bits.astype(bool)
        N, L = self.N, self.L

        if L == 1:
            return sc_decode_recursive(llr, frozen), 0.0

        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=int)}]
        for phi in range(N):
            expanded = []
            for path in paths:
                llr_phi = _llr_at_phase(llr, frozen, path["u"], phi)
                if frozen[phi]:
                    pen = abs(llr_phi) if llr_phi < 0 else 0.0
                    expanded.append({"pm": path["pm"] + pen, "u": path["u"]})
                else:
                    for bit in (0, 1):
                        pen = 0.0
                        if (bit == 0 and llr_phi < 0) or (bit == 1 and llr_phi >= 0):
                            pen = abs(llr_phi)
                        nu = path["u"].copy()
                        nu[phi] = bit
                        expanded.append({"pm": path["pm"] + pen, "u": nu})
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[:L]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_idx = np.where(~frozen)[0]
            valid = [p for p in paths if crc_check(p["u"][info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
        return best["u"], best["pm"]
