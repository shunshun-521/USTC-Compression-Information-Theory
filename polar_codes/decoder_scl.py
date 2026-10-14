"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode, _SCNode, _assign_lanes, _recursive_decode_node


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, width):
    reg = 0
    for b in np.asarray(bits, dtype=int):
        if width == 8:
            reg ^= int(b) << 7
            for _ in range(8):
                reg = ((reg << 1) ^ poly) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
        else:
            reg ^= int(b) << 15
            reg = ((reg << 1) ^ poly) & 0xFFFF if (reg & 0x8000) else (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_process(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    elif crc_length == 16:
        rem = _crc_process(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return _crc_process(bits, _CRC8_POLY, 8) == 0
    if crc_length == 16:
        return _crc_process(bits, _CRC16_POLY, 16) == 0
    return False


def _llr_at_phase(llr_ch, frozen_bits, decided_bits, phi):
    """给定前缀 decided_bits[:phi]，计算相位 phi 的 LLR。"""
    N = len(llr_ch)
    root = _SCNode(N)
    root.lam[:] = llr_ch
    _assign_lanes(root)

    def mark(node):
        if node.is_leaf:
            lane = node.lane
            if lane < phi:
                node.frozen = True
                node.s[0] = int(decided_bits[lane])
            else:
                node.frozen = bool(frozen_bits[lane])
        else:
            mark(node.left)
            mark(node.right)

    mark(root)
    _recursive_decode_node(root)

    def get_leaf_lam(node):
        if node.is_leaf:
            return node.lam[0] if node.lane == phi else None
        a = get_leaf_lam(node.left)
        return a if a is not None else get_leaf_lam(node.right)

    return float(get_leaf_lam(root))


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, np.zeros(self.N, dtype=int))]

        for phi in range(self.N):
            expanded = []
            for pm, bits in paths:
                llr = _llr_at_phase(llr_ch, self.frozen_bits, bits, phi)
                if self.frozen_bits[phi]:
                    v = 0
                    pm2 = pm + (abs(llr) if llr < 0 else 0.0)
                    nb = bits.copy()
                    nb[phi] = 0
                    expanded.append((pm2, nb))
                else:
                    for u in (0, 1):
                        pm2 = pm + (0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr))
                        nb = bits.copy()
                        nb[phi] = u
                        expanded.append((pm2, nb))
            expanded.sort(key=lambda x: x[0])
            paths = expanded[: self.list_size]

        paths.sort(key=lambda x: x[0])
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for pm, bits in paths:
                if crc_check(bits[info_idx], self.crc_length):
                    return bits, pm
        return paths[0][1], paths[0][0]
