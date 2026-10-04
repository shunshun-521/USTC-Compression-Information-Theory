"""
极化码 SCL（串行抵消列表）译码器
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _B_check, _s_updater, _Li


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_len):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_len - 1)
        for _ in range(8 if crc_len <= 8 else 16):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("llrs", "s", "pm", "u_hat")

    def __init__(self, n, N, llr_ch):
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.llrs[n, :] = llr_ch
        self.s = -np.ones((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_metric_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if hard == bit else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        paths = [_Path(self.n, self.N, llr_ch) for _ in range(L)]
        paths[0].pm = 0.0
        for p in paths[1:]:
            p.pm = np.inf

        for ii in range(self.N):
            candidates = []
            for pi, path in enumerate(paths):
                if np.isinf(path.pm):
                    continue
                llr_bit = _Li(0, ii, path.llrs, path.s, self.n)
                path.llrs[0, ii] = llr_bit
                if self.frozen_bits[ii]:
                    pen = self._path_metric_penalty(llr_bit, 0)
                    path.pm += pen
                    path.u_hat[ii] = 0
                    path.s[0, ii] = 0
                    path.llrs[0, ii] = np.inf
                    candidates.append((path.pm, pi, None))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._path_metric_penalty(llr_bit, bit)
                        candidates.append((new_pm, pi, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            used = set()
            for pm, parent_idx, bit in candidates:
                if len(new_paths) >= L:
                    break
                parent = paths[parent_idx]
                key = (parent_idx, bit)
                if bit is not None and key in used:
                    continue
                if bit is None:
                    new_paths.append(parent)
                    parent.pm = pm
                    continue
                used.add(key)
                child = _Path(self.n, self.N, llr_ch)
                child.llrs = np.copy(parent.llrs)
                child.s = np.copy(parent.s)
                child.u_hat = parent.u_hat.copy()
                child.pm = pm
                child.u_hat[ii] = bit
                child.s[0, ii] = bit
                new_paths.append(child)

            while len(new_paths) < L:
                new_paths.append(_Path(self.n, self.N, llr_ch))
                new_paths[-1].pm = np.inf
            paths = new_paths[:L]

        valid = []
        for path in paths:
            if np.isinf(path.pm):
                continue
            if self.crc_length > 0:
                payload = path.u_hat[self.info_indices]
                if not crc_check(payload, self.crc_length):
                    continue
            valid.append(path)

        if self.crc_length > 0 and valid:
            best = min(valid, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), float(best.pm)
