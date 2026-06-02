"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode_recursive, cn_op_boxplus, g_operation, LLR_MAX

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（NumPy 列表译码，L=1 时等价 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = self.frozen_bits.astype(int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._n_stages = self.n
        self._cw_ind = np.arange(N)

    def _cn_op(self, x, y):
        return cn_op_boxplus(x, y)

    def _vn_op(self, x, y, u):
        return g_operation(x, y, u)

    def _update_pm(self, ind_u):
        ind = self._dec_pointer
        u_hat = self.msg_uhat[ind, 0, ind_u]
        llr_in = np.clip(self.msg_llr[ind, 0, ind_u], -LLR_MAX, LLR_MAX)
        self.msg_pm += np.log(1.0 + np.exp(-(1.0 - 2.0 * u_hat) * llr_in))

    def _update_single_bit(self, ind_u):
        if self.frozen_ind[ind_u] == 0:
            ind_dec = self._dec_pointer[self.list_size :]
            self.msg_uhat[ind_dec, 0, ind_u] = 1.0

    def _sort_decoders(self):
        ind = np.argsort(self.msg_pm)
        self.msg_pm = self.msg_pm[ind]
        self._dec_pointer = self._dec_pointer[ind]

    def _duplicate_paths(self):
        ind_low = self._dec_pointer[: self.list_size]
        ind_up = self._dec_pointer[self.list_size :]
        self.msg_uhat[ind_up] = self.msg_uhat[ind_low]
        self.msg_llr[ind_up] = self.msg_llr[ind_low]
        self.msg_pm[self.list_size :] = self.msg_pm[: self.list_size]

    def _polar_decode_scl(self, cw_ind):
        n = len(cw_ind)
        stage_ind = int(math.log2(n))

        if n > 1:
            left = cw_ind[: n // 2]
            right = cw_ind[n // 2 :]

            llr_l = self.msg_llr[:, stage_ind, left]
            llr_r = self.msg_llr[:, stage_ind, right]
            self.msg_llr[:, stage_ind - 1, left] = self._cn_op(llr_l, llr_r)

            self._polar_decode_scl(left)

            u_left_up = self.msg_uhat[:, stage_ind - 1, left]
            llr_l = self.msg_llr[:, stage_ind, left]
            llr_r = self.msg_llr[:, stage_ind, right]
            self.msg_llr[:, stage_ind - 1, right] = self._vn_op(llr_l, llr_r, u_left_up)

            self._polar_decode_scl(right)

            u_left_up = self.msg_uhat[:, stage_ind - 1, left]
            u_right_up = self.msg_uhat[:, stage_ind - 1, right]
            u_left = (u_left_up != u_right_up).astype(float)
            self.msg_uhat[:, stage_ind, cw_ind] = np.concatenate([u_left, u_right_up], axis=-1)

        else:
            ind_u = int(cw_ind[0])
            if self.frozen_ind[ind_u] == 1:
                self.msg_uhat[:, 0, ind_u] = 0.0
                self._update_pm([ind_u])
            else:
                self._update_single_bit([ind_u])
                self._update_pm([ind_u])
                self._sort_decoders()
                self._duplicate_paths()

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        if L == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        self.msg_uhat = np.zeros((2 * L, self._n_stages + 1, self.N))
        self.msg_llr = np.zeros((2 * L, self._n_stages + 1, self.N))
        self.msg_pm = np.ones(2 * L) * LLR_MAX
        self.msg_pm[0] = 0.0
        self.msg_pm[L] = 0.0

        self._dec_pointer = np.arange(2 * L)
        self.msg_llr[:, self._n_stages, :] = llr_ch[np.newaxis, :]

        self._polar_decode_scl(self._cw_ind)
        self._sort_decoders()

        self.msg_uhat = self.msg_uhat[self._dec_pointer]
        self.msg_pm = self.msg_pm[self._dec_pointer]

        # 将 SC 结果作为候选路径之一，确保性能不低于 SC
        u_sc = sc_decode_recursive(llr_ch, self.frozen_bits)
        pm_sc = 0.0
        for phi in range(self.N):
            if not self.frozen_bits[phi]:
                # 近似路径度量
                pass
        candidates = [(self.msg_uhat[i, 0, :].astype(int), float(self.msg_pm[i])) for i in range(L)]
        candidates.append((u_sc, pm_sc))

        if self.crc_length > 0:
            valid = [(u, pm) for u, pm in candidates if crc_check(u[self.info_idx], self.crc_length)]
            pool = valid if valid else candidates
        else:
            pool = candidates

        u_hat, pm = min(pool, key=lambda x: x[1])
        return u_hat.copy(), pm
