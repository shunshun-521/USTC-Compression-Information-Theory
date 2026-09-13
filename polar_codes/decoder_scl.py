"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _cn_op_exact, _frozen_to_ind, _vn_op_exact
from encoder import bit_reversal_permutation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_bits_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_bits_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _u_up_from_decoded(u_ref, llr, offset=0):
    """根据已知译码结果计算子树 u_up（与 SC 递归返回值一致）"""

    def _rec(node_llr, offset_local):
        n = len(node_llr)
        if n == 1:
            return np.array([float(u_ref[offset_local])])
        half = n // 2
        u1_up = _rec(_cn_op_exact(node_llr[:half], node_llr[half:]), offset_local)
        u2_up = _rec(
            _vn_op_exact(node_llr[:half], node_llr[half:], u1_up),
            offset_local + half,
        )
        u1_up_new = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
        return np.concatenate([u1_up_new, u2_up])

    return _rec(llr, offset)


def _llr_at_index(llr, frozen, u_known, target):
    """计算 target 位置比特的 LLR（已知 u_known[0:target]）"""

    def _rec(node_llr, offset):
        n = len(node_llr)
        if n == 1:
            return node_llr[0]
        half = n // 2
        if target < offset + half:
            return _rec(_cn_op_exact(node_llr[:half], node_llr[half:]), offset)
        u_up = _u_up_from_decoded(u_known, node_llr[:half], offset)
        return _rec(
            _vn_op_exact(node_llr[:half], node_llr[half:], u_up),
            offset + half,
        )

    return _rec(llr, 0)


class SCLDecoder:
    """逐比特 SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_ind = _frozen_to_ind(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _llr_at_index(llr, self.frozen_ind, path["u"], phi)
                if self.frozen_bits[phi] == 1:
                    new_u = path["u"].copy()
                    new_u[phi] = 0
                    candidates.append(
                        {
                            "pm": path["pm"] + _pm_penalty(llr_phi, 0),
                            "u": new_u,
                        }
                    )
                else:
                    for bit in (0, 1):
                        new_u = path["u"].copy()
                        new_u[phi] = bit
                        candidates.append(
                            {
                                "pm": path["pm"] + _pm_penalty(llr_phi, bit),
                                "u": new_u,
                            }
                        )
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].copy(), best["pm"]
