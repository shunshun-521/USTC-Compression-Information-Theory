"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    INF,
    f_operation,
    g_operation,
    _b_check,
    _s_updater,
    _li,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int)
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = np.zeros(crc_length, dtype=int)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, llrs, s, pm, u_hat):
        return llrs.copy(), s.copy(), pm, u_hat.copy()

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = []
        s_list = []
        pm_list = []
        u_hat_list = []

        llrs = np.full((n + 1, N), INF, dtype=np.float64)
        llrs[n, :] = llr_ch.copy()
        s = np.full((n + 1, N), -1, dtype=int)
        llrs_list.append(llrs)
        s_list.append(s)
        pm_list.append(0.0)
        u_hat_list.append(np.zeros(N, dtype=int))

        for idx in range(N):
            candidates = []

            for path_id in range(len(llrs_list)):
                llrs = llrs_list[path_id]
                s = s_list[path_id]
                pm = pm_list[path_id]
                u_hat = u_hat_list[path_id]

                llr_val = _li(0, idx, llrs, s, n)

                if self.frozen_bits[idx]:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_llrs, new_s, new_pm, new_u = self._copy_path(llrs, s, pm, u_hat)
                    new_pm += penalty
                    new_u[idx] = 0
                    new_s[0, idx] = 0
                    candidates.append((new_pm, new_llrs, new_s, new_u))
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
                        new_llrs, new_s, new_pm, new_u = self._copy_path(llrs, s, pm, u_hat)
                        new_pm += penalty
                        new_u[idx] = bit
                        new_s[0, idx] = bit
                        candidates.append((new_pm, new_llrs, new_s, new_u))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[:L]

            llrs_list = [c[1] for c in selected]
            s_list = [c[2] for c in selected]
            pm_list = [c[0] for c in selected]
            u_hat_list = [c[3] for c in selected]

        if self.crc_length > 0:
            info_positions = np.where(self.frozen_bits == 0)[0]
            crc_pass = [
                (pm_list[i], u_hat_list[i])
                for i in range(len(u_hat_list))
                if crc_check(u_hat_list[i][info_positions], self.crc_length)
            ]
            if crc_pass:
                best_pm, best_u = min(crc_pass, key=lambda x: x[0])
                return best_u, best_pm

        best_idx = int(np.argmin(pm_list))
        return u_hat_list[best_idx], pm_list[best_idx]
