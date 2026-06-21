"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _llr_to_work, _hard_decision


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    hard = _hard_decision(llr)
    return 0.0 if bit == hard else abs(llr)


def _llr_at_index(llr_root, u_prefix, phi):
    """
    已知 u[0:phi] 时，计算 u[phi] 处的 LLR。
  u_prefix 为长度 phi 的数组。
    """

    def recurse(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            return llr_node[0]
        half = n // 2
        if phi < offset + half:
            return recurse(f_operation(llr_node[:half], llr_node[half:]), offset)
        u_left = u_prefix[offset:offset + half].astype(np.float64)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        return recurse(llr_right, offset + half)

    return recurse(llr_root, 0)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = _llr_to_work(llr_ch)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _llr_at_index(llr, path["u"][:phi], phi)
                if self.frozen_bits[phi]:
                    pm = path["pm"] + _pm_penalty(llr_phi, 0)
                    u_new = path["u"].copy()
                    u_new[phi] = 0
                    candidates.append({"pm": pm, "u": u_new})
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + _pm_penalty(llr_phi, bit)
                        u_new = path["u"].copy()
                        u_new[phi] = bit
                        candidates.append({"pm": pm, "u": u_new})
            candidates.sort(key=lambda x: x["pm"])
            paths = [{"pm": c["pm"], "u": c["u"]} for c in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        return best["u"], best["pm"]
