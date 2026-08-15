"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, f_boxplus


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_length):
    """计算 CRC 余数。"""
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    expected = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i in range(crc_length):
        received |= int(bits[-crc_length + i]) << (crc_length - 1 - i)
    return expected == received


# ==================== SCL 译码器 ====================

class _Path:
  __slots__ = ("pm", "L", "B", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.pm = 0.0
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=int)
      self.L[:, 0] = llr_ch
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    @staticmethod
    def _bit_reversed(i, n):
        result = 0
        for bit in range(n):
            if i & (1 << bit):
                result |= 1 << (n - 1 - bit)
        return result

    @staticmethod
    def _active_llr_level(i, n):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    @staticmethod
    def _active_bit_level(i, n):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def _update_llrs(self, path, l):
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self._bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    candidates.append((path.pm + penalty, path, 0))
                else:
                    for bit in (0, 1):
                        penalty = self._path_metric_penalty(llr, bit)
                        candidates.append((path.pm + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child = _Path(self.N, self.n, llr_ch)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()

                child.B[l, self.n] = 0 if self.frozen_bits[l] else bit
                child.u_hat[l] = child.B[l, self.n]
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for p in paths:
                check_bits = p.u_hat[self.info_indices] if self.info_indices is not None else p.u_hat
                if crc_check(check_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
