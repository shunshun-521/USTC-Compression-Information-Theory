"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_process(info_bits, poly, crc_length)
    for _ in range(crc_length):
        top = 1 << (crc_length - 1)
        mask = (1 << crc_length) - 1
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _llr_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int_)
        L0[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": L0, "B": B0})

        for i in range(N):
            l = int(f"{i:0{n}b}"[::-1], 2)
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    pen = self._llr_penalty(llr, 0)
                    p = {
                        "pm": path["pm"] + pen,
                        "L": path["L"],
                        "B": path["B"].copy(),
                    }
                    p["B"][l, n] = 0
                    self._update_bits(p["B"], l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = {
                            "pm": path["pm"] + self._llr_penalty(llr, u),
                            "L": path["L"],
                            "B": path["B"].copy(),
                        }
                        p["B"][l, n] = u
                        self._update_bits(p["B"], l)
                        candidates.append(p)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]
            for p in paths:
                p["L"] = p["L"].copy()

        best_crc = None
        best_pm = float("inf")
        best_any = min(paths, key=lambda p: p["pm"])

        for p in paths:
            u_hat = p["B"][:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and p["pm"] < best_pm:
                    best_pm = p["pm"]
                    best_crc = u_hat
            if p["pm"] < best_any["pm"]:
                best_any = p

        if best_crc is not None:
            return best_crc.copy(), best_pm
        return best_any["B"][:, n].astype(int).copy(), best_any["pm"]
