"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    """按 MSB 优先对 data_bits 做 CRC 除法，返回余数比特"""
    reg = [0] * crc_len
    for bit in data_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= fb
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    data = bits[:-crc_length]
    rem = _crc_divide(data, poly, crc_length)
    return np.array_equal(rem, bits[-crc_length:])


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, u):
        """路径度量更新：与 LLR 符号一致不惩罚，否则加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        if u != hard:
            return pm + abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 最优路径估计
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        Lsz = self.L_size

        # 每条路径: LLR 矩阵、比特矩阵、路径度量、父路径索引（lazy copy）
        paths_L = [np.zeros((N, n + 1), dtype=np.float64) for _ in range(Lsz)]
        paths_B = [np.zeros((N, n + 1), dtype=np.int8) for _ in range(Lsz)]
        paths_pm = [0.0]
        active = 1

        paths_L[0][:, 0] = llr_ch
        u_hat_all = [np.zeros(N, dtype=int) for _ in range(Lsz)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for p in range(active):
                Lm = paths_L[p]
                Bm = paths_B[p]
                pm0 = paths_pm[p]

                # 更新 LLR 树到比特 l
                for s in range(n - _active_llr_level(l, n), n):
                    block = 1 << (s + 1)
                    half = block // 2
                    for j in range(l, N, block):
                        if j % block < half:
                            Lm[j, s + 1] = f_operation(Lm[j, s], Lm[j + half, s])
                        else:
                            top_bit = Bm[j - half, s + 1]
                            Lm[j, s + 1] = g_operation(
                                Lm[j - half, s], Lm[j, s], top_bit
                            )

                cur_llr = Lm[l, n]

                if self.frozen_bits[l]:
                    u_val = 0
                    new_pm = self._path_metric_update(pm0, cur_llr, u_val)
                    Bm[l, n] = 0
                    u_hat_all[p][l] = 0
                    candidates.append((new_pm, p, u_val))
                else:
                    for u_val in (0, 1):
                        new_pm = self._path_metric_update(pm0, cur_llr, u_val)
                        candidates.append((new_pm, p, u_val))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:Lsz]

            new_active = len(candidates)
            new_paths_L = []
            new_paths_B = []
            new_paths_pm = []
            new_u_hat = []

            for new_pm, parent, u_val in candidates:
                if len(new_paths_L) < len(paths_L) and len(new_paths_L) < active:
                    idx = len(new_paths_L)
                    Lm = paths_L[idx]
                    Bm = paths_B[idx]
                    uh = u_hat_all[idx]
                else:
                    Lm = paths_L[len(new_paths_L)].copy()
                    Bm = paths_B[len(new_paths_L)].copy()
                    uh = u_hat_all[len(new_paths_L)].copy()
                    idx = len(new_paths_L)

                if parent < active:
                    Lm[:] = paths_L[parent]
                    Bm[:] = paths_B[parent]
                    uh[:] = u_hat_all[parent]

                l = bit_reversed(i, n)
                Lm[l, n]  # ensure computed
                cur_llr = Lm[l, n]
                Bm[l, n] = u_val
                uh[l] = u_val

                if l >= N // 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block = 1 << s
                        half = block // 2
                        for j in range(l, -1, -block):
                            if j % block >= half:
                                Bm[j - half, s - 1] = Bm[j, s] ^ Bm[j - half, s]
                                Bm[j, s - 1] = Bm[j, s]

                new_paths_L.append(Lm)
                new_paths_B.append(Bm)
                new_paths_pm.append(new_pm)
                new_u_hat.append(uh)

            paths_L = new_paths_L
            paths_B = new_paths_B
            paths_pm = new_paths_pm
            u_hat_all = new_u_hat
            active = new_active

        # 选择最优路径
        best_idx = 0
        crc_pass = []
        for p in range(active):
            if self.crc_length > 0:
                info_bits = u_hat_all[p][self.info_indices]
                if len(info_bits) >= self.crc_length:
                    if crc_check(info_bits, self.crc_length):
                        crc_pass.append(p)
            if paths_pm[p] < paths_pm[best_idx]:
                best_idx = p

        if self.crc_length > 0 and crc_pass:
            best_idx = min(crc_pass, key=lambda p: paths_pm[p])

        return u_hat_all[best_idx], paths_pm[best_idx]
