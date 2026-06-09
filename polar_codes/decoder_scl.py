"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _compute_llr, _s_updater, INF


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    if crc_length == 16:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8)
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    r = len(poly) - 1
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[len(info_bits):len(info_bits) + crc_length]])


def crc_check(bits, crc_length=8):
    """检验 bits 尾部 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    payload = bits[:-crc_length]
    msg = np.concatenate([payload, np.zeros(crc_length, dtype=int)])
    for i in range(len(payload)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    expected = msg[len(payload):len(payload) + crc_length]
    return np.array_equal(expected, bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        N = self.N
        n = self.n

        llrs = [np.full((n + 1, N), -INF, dtype=np.float64) for _ in range(L)]
        bits = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
        for path in range(L):
            llrs[path][n, :] = llr_ch

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        active = 1

        for phi in range(N):
            dm = np.zeros(L, dtype=np.float64)

            if self.frozen_bits[phi]:
                for path in range(active):
                    llrs[path][0, phi] = _compute_llr(0, phi, llrs[path], bits[path])
                    bits[path][0, phi] = 0
                    pm[path] += abs(min(llrs[path][0, phi], 0.0))
            else:
                for path in range(active):
                    llrs[path][0, phi] = _compute_llr(0, phi, llrs[path], bits[path])
                    bits[path][0, phi] = 1 if llrs[path][0, phi] < 0 else 0
                    dm[path] = abs(llrs[path][0, phi])

            if (not self.frozen_bits[phi]) and L > 1:
                candidates = []
                for path in range(active):
                    bit0 = bits[path][0, phi]
                    pm0 = pm[path]
                    candidates.append((pm0, path, bit0))
                    candidates.append((pm0 + dm[path], path, 1 - bit0))

                candidates.sort(key=lambda x: x[0])
                selected = candidates[:L]
                while len(selected) < L:
                    selected.append(selected[-1])

                new_llrs = []
                new_bits = []
                new_pm = np.zeros(L, dtype=np.float64)
                for new_idx, (metric, src, bit_val) in enumerate(selected):
                    new_llrs.append(llrs[src].copy())
                    new_bits.append(bits[src].copy())
                    new_bits[new_idx][0, phi] = bit_val
                    new_pm[new_idx] = metric

                llrs = new_llrs
                bits = new_bits
                pm = new_pm
                active = len(llrs)

        best_paths = list(range(L))
        if self.crc_length > 0:
            passed = []
            for path in range(L):
                info_bits = bits[path][0, self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(path)
            if passed:
                best_paths = passed

        best = min(best_paths, key=lambda p: pm[p])
        u_hat = bits[best][0, :].astype(int)
        return u_hat, float(pm[best])
