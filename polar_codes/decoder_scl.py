"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive
from encoder import polar_encode


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
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _llr_at_phase(channel_llr, frozen_bits, u_prefix, phi):
    """计算第 phi 个比特的 SC LLR（u_prefix[0:phi] 已固定）"""
    u_prefix = np.asarray(u_prefix, dtype=np.int8)

    def llr_sub(llrs, indices):
        if len(indices) == 1:
            return float(llrs[0])
        half = len(indices) // 2
        left_idx = indices[:half]
        right_idx = indices[half:]
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        if phi >= indices[half]:
            u_left = [int(u_prefix[idx]) for idx in left_idx]
            right_llr = np.array(
                [g_operation(llrs[i], llrs[i + half], u_left[i]) for i in range(half)],
                dtype=np.float64,
            )
            return llr_sub(right_llr, right_idx)
        return llr_sub(left_llr, left_idx)

    return llr_sub(channel_llr, list(range(len(channel_llr))))


class SCLDecoder:
    """SCL 译码器（按相位扩展路径）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_add(self, pm, llr, u):
        u_hat = 0 if llr >= 0 else 1
        if u != u_hat:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]
        for phi in range(self.N):
            llr_phi_cache = {}
            candidates = []
            for p in paths:
                key = tuple(p["u"][:phi])
                if key not in llr_phi_cache:
                    llr_phi_cache[key] = _llr_at_phase(llr_ch, self.frozen_bits, p["u"], phi)
                llr = llr_phi_cache[key]
                if self.frozen_bits[phi]:
                    pm = self._pm_add(p["pm"], llr, 0)
                    u_new = p["u"].copy()
                    u_new[phi] = 0
                    candidates.append({"pm": pm, "u": u_new})
                else:
                    for bit in (0, 1):
                        pm = self._pm_add(p["pm"], llr, bit)
                        u_new = p["u"].copy()
                        u_new[phi] = bit
                        candidates.append({"pm": pm, "u": u_new})
            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        def codeword_consistent(u_vec):
            x = polar_encode(u_vec)
            return np.all((x == 0) == (llr_ch >= 0))

        best = None
        if self.crc_length > 0:
            for p in paths:
                if crc_check(p["u"], self.crc_length):
                    if best is None or p["pm"] < best["pm"]:
                        best = p
        if best is None:
            valid = [p for p in paths if codeword_consistent(p["u"])]
            pool = valid if valid else paths
            best = min(pool, key=lambda p: (p["pm"], p["u"].tobytes()))
        return best["u"], best["pm"]
