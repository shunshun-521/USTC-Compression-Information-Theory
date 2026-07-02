"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from channel import prepare_channel_llr
from decoder_sc import _lazy_llr, g_operation, f_operation


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
)


def _crc_remainder(bits, poly):
    msg = np.concatenate([bits, np.zeros(len(poly) - 1, dtype=int)])
    for i in range(len(bits)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return msg[-(len(poly) - 1):]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], _crc_remainder(payload, poly))


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = prepare_channel_llr(llr_ch)
        N, n, L = self.N, self.n, self.list_size

        llrs = [
            np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)
        ]
        bits = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
        for l_idx in range(L):
            llrs[l_idx][n, :] = llr_ch

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        active = [True] * L

        for phi in range(N):
            candidates = []

            for l_idx in range(L):
                if not active[l_idx]:
                    continue

                if self.frozen_bits[phi]:
                    llrs[l_idx][0, phi] = _lazy_llr(0, phi, llrs[l_idx], bits[l_idx])
                    bits[l_idx][0, phi] = 0
                    penalty = 0.0 if llrs[l_idx][0, phi] >= 0 else abs(llrs[l_idx][0, phi])
                    candidates.append((pm[l_idx] + penalty, l_idx, 0))
                else:
                    llrs[l_idx][0, phi] = _lazy_llr(0, phi, llrs[l_idx], bits[l_idx])
                    llr0 = llrs[l_idx][0, phi]
                    for bit in (0, 1):
                        penalty = 0.0 if (llr0 >= 0) == (bit == 0) else abs(llr0)
                        candidates.append((pm[l_idx] + penalty, l_idx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]

            new_llrs = [
                np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)
            ]
            new_bits = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
            new_pm = np.full(L, np.inf, dtype=np.float64)
            new_active = [False] * L
            new_u_partial = [None] * L

            for i, (new_pm_val, parent, bit) in enumerate(candidates):
                new_llrs[i] = llrs[parent].copy()
                new_bits[i] = bits[parent].copy()
                new_pm[i] = new_pm_val
                new_active[i] = True
                new_bits[i][0, phi] = 0 if self.frozen_bits[phi] else bit
                new_llrs[i][0, phi] = llrs[parent][0, phi]

            llrs, bits, pm, active = new_llrs, new_bits, new_pm, new_active

        paths_u = []
        paths_pm = []
        for l_idx in range(L):
            if not active[l_idx]:
                continue
            u_hat = bits[l_idx][0, :].astype(int)
            paths_u.append(u_hat)
            paths_pm.append(pm[l_idx])

        if not paths_u:
            return np.zeros(N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = []
            for u_hat, path_pm in zip(paths_u, paths_pm):
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((path_pm, u_hat))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])
                return best_u.copy(), best_pm

        best_idx = int(np.argmin(paths_pm))
        return paths_u[best_idx].copy(), float(paths_pm[best_idx])
