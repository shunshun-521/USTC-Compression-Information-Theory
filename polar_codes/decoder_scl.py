"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode, _get_G


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


def _llr_at_phi(llr, u_prefix, phi):
    """已知 u[0:phi] 时计算比特 phi 的 LLR。"""

    def decode_left(lam, off, length, limit):
        """译码段内索引 < limit 的比特，返回整段 u_seg。"""
        if length == 1:
            if off < limit:
                return np.array([u_prefix[off]], dtype=np.int8)
            return np.array([0 if lam[0] >= 0 else 1], dtype=np.int8)
        half = length // 2
        lam_l = f_operation(lam[:half], lam[half:])
        u_l = decode_left(lam_l, off, half, limit)
        lam_r = g_operation(lam[:half], lam[half:], u_l)
        u_r = decode_left(lam_r, off + half, half, limit)
        u_seg = np.zeros(length, dtype=np.int8)
        for i in range(half):
            u_seg[i] = u_l[i] ^ u_r[i]
            u_seg[half + i] = u_r[i]
        return u_seg

    def rec(lam, off, length):
        if length == 1:
            return float(lam[0])
        half = length // 2
        if phi < off + half:
            lam_l = f_operation(lam[:half], lam[half:])
            return rec(lam_l, off, half)
        lam_l = f_operation(lam[:half], lam[half:])
        u_l = decode_left(lam_l, off, half, phi)
        lam_r = g_operation(lam[:half], lam[half:], u_l)
        return rec(lam_r, off + half, half)

    return rec(np.asarray(llr, dtype=np.float64), 0, len(llr))


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, list_size)
        self.crc_length = crc_length
        self.G = _get_G(N)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            u = sc_decode(llr, self.frozen)
            return u, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            candidates = []
            for st in paths:
                bit_llr = _llr_at_phi(llr, st["u"], phi)
                if self.frozen[phi]:
                    pen = 0.0 if bit_llr >= 0 else abs(bit_llr)
                    st2 = {"pm": st["pm"] + pen, "u": st["u"].copy()}
                    st2["u"][phi] = 0
                    candidates.append(st2)
                else:
                    for bit in (0, 1):
                        pen = 0.0
                        if (bit == 0 and bit_llr < 0) or (bit == 1 and bit_llr >= 0):
                            pen = abs(bit_llr)
                        u2 = st["u"].copy()
                        u2[phi] = bit
                        candidates.append({"pm": st["pm"] + pen, "u": u2})
            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.L]

        if self.crc_length > 0:
            info_mask = ~self.frozen
            valid = []
            for st in paths:
                info_bits = st["u"][info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(st)
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        u_hat = best["u"].copy()
        u_hat[self.frozen] = 0
        return u_hat, best["pm"]
