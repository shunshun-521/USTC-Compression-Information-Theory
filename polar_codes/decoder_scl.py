"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int32)
    else:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int32)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int32)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（比特倒序相位，Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.phases = [bit_reversed(i, self.n) for i in range(N)]

    def _branch_pm(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if u == hard else abs(llr)
        return pm + penalty

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        start_s = n - _active_bit_level(l, n)
        for s in range(n, start_s, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = llr_ch
        B0 = np.zeros((N, n + 1), dtype=np.int32)

        paths = [{"pm": 0.0, "L": L0, "B": B0}]

        for l in self.phases:
            candidates = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = self._branch_pm(path["pm"], llr, 0)
                    B_new = path["B"].copy()
                    B_new[l, n] = 0
                    self._update_bits(B_new, l)
                    candidates.append({"pm": pm, "L": path["L"].copy(), "B": B_new})
                else:
                    for u in (0, 1):
                        pm = self._branch_pm(path["pm"], llr, u)
                        B_new = path["B"].copy()
                        B_new[l, n] = u
                        self._update_bits(B_new, l)
                        candidates.append(
                            {"pm": pm, "L": path["L"].copy(), "B": B_new}
                        )

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path["B"][:, n].astype(np.int32)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(np.int32), best["pm"]
