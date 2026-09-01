"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    for trial in range(1 << crc_length):
        crc_bits = np.array([(trial >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
        full = np.concatenate([info_bits, crc_bits])
        if _crc_remainder(full, poly, crc_length) == 0:
            return full
    raise RuntimeError("CRC 编码失败")


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    def __init__(self, N, n, llr_ch, frozen_bits):
        self.N, self.n = N, n
        self.pm = 0.0
        self.frozen_bits = frozen_bits
        self.u_hat = np.zeros(N, dtype=int)

        self.bits = np.zeros((N + 1, n + 2), dtype=np.int8)
        self.bits_updated = np.zeros((N + 1, n + 2), dtype=bool)
        self.bits_updated[1:N + 1, 1] = frozen_bits

        self.llrs = np.zeros((N + 1, n + 2))
        self.llrs[1:N + 1, n + 1] = llr_ch
        self.llrs_updated = np.zeros((N + 1, n + 2), dtype=bool)
        self.llrs_updated[1:N + 1, n + 1] = True

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.N, p.n = self.N, self.n
        p.pm = self.pm
        p.frozen_bits = self.frozen_bits
        p.u_hat = self.u_hat.copy()
        p.bits = self.bits.copy()
        p.bits_updated = self.bits_updated.copy()
        p.llrs = self.llrs.copy()
        p.llrs_updated = self.llrs_updated.copy()
        return p

    def _update_bit(self, row, col):
        if self.bits_updated[row, col]:
            return
        offset = max(1, self.N // (2 ** (self.n + 2 - col)))
        if (row - 1) % (2 * offset) >= offset:
            if not self.bits_updated[row, col - 1]:
                self._update_bit(row, col - 1)
            self.bits[row, col] = self.bits[row, col - 1]
        else:
            if not self.bits_updated[row, col - 1]:
                self._update_bit(row, col - 1)
            if not self.bits_updated[row + offset, col - 1]:
                self._update_bit(row + offset, col - 1)
            self.bits[row, col] = self.bits[row, col - 1] ^ self.bits[row + offset, col - 1]
        self.bits_updated[row, col] = True

    def _update_llr(self, row, col):
        if col > self.n + 1 or self.llrs_updated[row, col]:
            return
        offset = max(1, self.N // (2 ** (self.n + 1 - col)))
        if (row - 1) % (2 * offset) >= offset:
            if not self.bits_updated[row - offset, col]:
                self._update_bit(row - offset, col)
            if not self.llrs_updated[row - offset, col + 1]:
                self._update_llr(row - offset, col + 1)
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            u = self.bits[row - offset, col]
            self.llrs[row, col] = g_operation(
                self.llrs[row - offset, col + 1], self.llrs[row, col + 1], u)
        else:
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            if not self.llrs_updated[row + offset, col + 1]:
                self._update_llr(row + offset, col + 1)
            self.llrs[row, col] = f_operation(
                self.llrs[row, col + 1], self.llrs[row + offset, col + 1])
        self.llrs_updated[row, col] = True

    def _pm_add(self, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            self.pm += abs(llr)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch, self.frozen_bits)]

        for i in range(1, self.N + 1):
            new_paths = []
            for path in paths:
                path._update_llr(i, 1)
                cur_llr = path.llrs[i, 1]
                if self.frozen_bits[i - 1]:
                    p = path.copy()
                    p._pm_add(cur_llr, 0)
                    p.u_hat[i - 1] = 0
                    p.bits[i, 1] = 0
                    p.bits_updated[i, 1] = True
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p._pm_add(cur_llr, u)
                        p.u_hat[i - 1] = u
                        p.bits[i, 1] = u
                        p.bits_updated[i, 1] = True
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            ok = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if ok:
                paths = ok

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
