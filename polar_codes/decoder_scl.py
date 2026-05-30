"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(bits, poly, width):
    """按比特串行 CRC-8/16（MSB 先行），返回寄存器终值"""
    reg = 0
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    nshift = 8 if width == 8 else 16
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(nshift):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC 编码：返回 [信息比特 | CRC]"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_bits(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length)
    crc = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """CRC 校验（与 crc_encode 一致）"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _bit_llr_at_phi(llr_ch, u_prefix, phi):
    """根据已译前缀 u_prefix[0:phi] 计算位置 phi 的 LLR"""

    def rec(llr, offset, length, target):
        if length == 1:
            return float(llr[0])
        h = length // 2
        if target < offset + h:
            return rec(f_operation(llr[:h], llr[h:]), offset, h, target)
        u_left = u_prefix[offset : offset + h].astype(np.float64)
        llr_r = g_operation(llr[:h], llr[h:], u_left)
        return rec(llr_r, offset + h, h, target)

    return rec(np.asarray(llr_ch, dtype=np.float64), 0, len(llr_ch), phi)


def _pm_add(pm, llr, u):
    """路径度量更新"""
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        SCL 译码。
        返回 (u_hat, pm)
        """
        from decoder_sc import sc_decode_recursive

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        if L == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for p in paths:
                llr = _bit_llr_at_phi(llr_ch, p["u"], phi)
                if self.frozen_bits[phi]:
                    u_new = p["u"].copy()
                    pm = p["pm"]
                    if llr < 0:
                        pm += abs(llr)
                    u_new[phi] = 0
                    candidates.append({"pm": pm, "u": u_new})
                else:
                    for u_bit in (0, 1):
                        u_new = p["u"].copy()
                        pm = _pm_add(p["pm"], llr, u_bit)
                        u_new[phi] = u_bit
                        candidates.append({"pm": pm, "u": u_new})
            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[:L]

        best = paths[0]
        if self.crc_length > 0:
            ok = [
                p
                for p in paths
                if crc_check(p["u"][self.info_idx], self.crc_length)
            ]
            if ok:
                best = min(ok, key=lambda x: x["pm"])

        return best["u"], float(best["pm"])
