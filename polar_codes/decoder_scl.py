"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, _frozen_bool, sc_decode
def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。多项式 CRC-8: 0x07, CRC-16: 0x8005"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
        top = 0x80
        mask = 0xFF
    elif crc_length == 16:
        poly = 0x8005
        top = 0x8000
        mask = 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _bit_llr(llr_node, u_prefix, phi, offset):
    """计算当前比特 phi 的 LLR（已知 u_prefix[0:phi]）"""
    n = len(llr_node)
    if n == 1:
        return float(llr_node[0])
    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    if phi < offset + half:
        return _bit_llr(llr_left, u_prefix, phi, offset)
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_prefix[offset : offset + half])
    return _bit_llr(llr_right, u_prefix, phi, offset + half)


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（按比特递推 LLR，路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = _frozen_bool(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                llr = _bit_llr(llr_ch, path["u"], phi, 0)
                if self.frozen_bits[phi]:
                    expanded.append(
                        {
                            "pm": path["pm"] + _pm_penalty(llr, 0),
                            "u": path["u"].copy(),
                        }
                    )
                    expanded[-1]["u"][phi] = 0
                else:
                    for bit in (0, 1):
                        u_new = path["u"].copy()
                        u_new[phi] = bit
                        expanded.append(
                            {"pm": path["pm"] + _pm_penalty(llr, bit), "u": u_new}
                        )

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.L]

        crc_ok = []
        for p in paths:
            info_bits = p["u"][self.info_positions]
            if self.crc_length == 0 or crc_check(info_bits, self.crc_length):
                crc_ok.append(p)

        best = min(crc_ok if crc_ok else paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
