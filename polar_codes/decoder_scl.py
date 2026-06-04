"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    f_operation,
    sc_decode_recursive,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        expected = crc_encode(bits[:-8], 8)[-8:]
    else:
        expected = crc_encode(bits[:-16], 16)[-16:]
    return np.array_equal(bits[-crc_length:], expected)


def _path_metric(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（多路径 L/B 矩阵，逐位比特倒序处理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        if L_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0})

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                _update_llrs(L, B, l, n, N)
                llr_bit = L[l, n]

                if l in self.frozen_set:
                    B[l, n] = 0
                    _update_bits(B, l, n, N)
                    new_paths.append(
                        {"L": L, "B": B, "pm": pm + _path_metric(llr_bit, 0)}
                    )
                else:
                    for u_val in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = u_val
                        _update_bits(Bc, l, n, N)
                        new_paths.append(
                            {
                                "L": Lc,
                                "B": Bc,
                                "pm": pm + _path_metric(llr_bit, u_val),
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        def path_to_u(path):
            return path["B"][:, n].astype(int)

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(path_to_u(p)[self.info_idx], self.crc_length)
            ]
            best = (
                min(valid, key=lambda p: p["pm"])
                if valid
                else min(paths, key=lambda p: p["pm"])
            )
        else:
            best = min(paths, key=lambda p: p["pm"])

        return path_to_u(best), best["pm"]
