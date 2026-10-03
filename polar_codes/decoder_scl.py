"""
极化码 SCL（串行抵消列表）译码器
Permuted SCD 因子图 + CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
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
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 L/B 数组引用）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br]

        paths = [
            {
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "pm": 0.0,
                "u": np.zeros(self.N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_perm

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    u = 0
                    pm = path["pm"] + self._pm_penalty(llr, u)
                    child = self._fork_path(path)
                    child["pm"] = pm
                    child["u"][l] = u
                    self._set_bit(child, l, u)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, u)
                        child = self._fork_path(path)
                        child["pm"] = pm
                        child["u"][l] = u
                        self._set_bit(child, l, u)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_ok = [p for p in paths if crc_check(p["u"][info_idx], self.crc_length)]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p["pm"])

        return best["u"].copy(), float(best["pm"])

    def _fork_path(self, path):
        return {
            "L": path["L"],
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u": path["u"].copy(),
        }

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _set_bit(self, path, l, u):
        B = path["B"]
        B[l, self.n] = u
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]
