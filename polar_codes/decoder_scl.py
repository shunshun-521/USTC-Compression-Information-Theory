"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from channel import permute_llr_for_decode
from decoder_sc import (
    SCDecoder,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = 0xFFFF
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class Path:
    """单条译码路径"""

    __slots__ = ("pm", "dec")

    def __init__(self, N, n):
        self.pm = 0.0
        self.dec = SCDecoder(N)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = permute_llr_for_decode(np.asarray(llr_ch, dtype=np.float64))
        N = self.N
        n = self.n
        L = self.list_size

        paths = [Path(N, n)]
        paths[0].dec.L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                path.dec.update_llrs(l)
                llr = path.dec.L[l, n]

                if l in self.frozen_set:
                    pm = path.pm + self._pm_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u in (0, 1):
                        pm = path.pm + self._pm_penalty(llr, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]

            new_paths = []
            for pm, parent, u_val in candidates:
                child = Path(N, n)
                child.pm = pm
                child.dec.L = parent.dec.L.copy()
                child.dec.B = parent.dec.B.copy()
                child.dec.B[l, n] = u_val
                child.dec.update_bits(l)
                new_paths.append(child)

            paths = new_paths

        best_path = None
        best_pm = float("inf")

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.dec.B[:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            search = valid if valid else paths
        else:
            search = paths

        for path in search:
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        u_hat = best_path.dec.B[:, n].astype(int)
        return u_hat, best_pm
