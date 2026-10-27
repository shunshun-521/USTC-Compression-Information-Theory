"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _li, _INF, _NEG_INF, f_operation, g_operation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_len):
    """对信息比特做 CRC 除法，返回余数比特（高位在前）"""
    reg = [0] * crc_len
    for bit in info_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= fb
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
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
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    payload = bits[:-crc_length]
    rem = _crc_divide(payload, poly, crc_length)
    return np.array_equal(rem, bits[-crc_length:])


# ==================== SCL 译码器 ====================


class _PathState:
    __slots__ = ("llrs", "bits", "pm")

    def __init__(self, n, N, llr_ch):
        self.llrs = np.full((n + 1, N), _NEG_INF, dtype=np.float64)
        self.llrs[n, :] = llr_ch
        self.bits = -np.ones((n + 1, N), dtype=np.int8)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 风格的路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        paths = [_PathState(n, N, llr_ch)]
        pms = np.zeros(L, dtype=np.float64)
        pms[1:] = np.inf
        active = 1

        for phi in range(N):
            dm = np.zeros(L, dtype=np.float64)
            for l in range(active):
                st = paths[l]
                llr_val = _li(0, phi, st.llrs, st.bits, n)
                st.llrs[0][phi] = llr_val
                if self.frozen_bits[phi]:
                    st.bits[0][phi] = 0
                    pms[l] += -llr_val * (llr_val < 0)
                else:
                    st.bits[0][phi] = 1 if llr_val < 0 else 0
                    dm[l] = abs(llr_val)

            if (not self.frozen_bits[phi]) and active < L:
                # 分裂路径
                while active < L:
                    parent = (active - 1) % max(active, 1)
                    new_state = _PathState(n, N, llr_ch)
                    new_state.llrs = paths[parent].llrs.copy()
                    new_state.bits = paths[parent].bits.copy()
                    new_state.bits[0][phi] = 1 - paths[parent].bits[0][phi]
                    paths.append(new_state)
                    pms[active] = pms[parent] + dm[parent]
                    active += 1

            if (not self.frozen_bits[phi]) and active >= 2:
                pm_dm = np.concatenate([pms[:active], pms[:active] + dm[:active]])
                order = np.argsort(pm_dm)
                keep = order[:L]
                new_paths = []
                new_pms = np.full(L, np.inf)
                for rank, idx in enumerate(keep):
                    if idx < active:
                        new_paths.append(paths[idx])
                        new_pms[rank] = pms[idx]
                    else:
                        src = idx - active
                        st = _PathState(n, N, llr_ch)
                        st.llrs = paths[src].llrs.copy()
                        st.bits = paths[src].bits.copy()
                        st.bits[0][phi] = 1 - paths[src].bits[0][phi]
                        new_paths.append(st)
                        new_pms[rank] = pms[src] + dm[src]
                paths = new_paths
                pms = new_pms
                active = min(L, len(paths))

        best = int(np.argmin(pms[:active]))
        u_hat = paths[best].bits[0, :].astype(int)
        u_hat[self.frozen_bits] = 0

        if self.crc_length > 0:
            K_info = len(self.info_indices) - self.crc_length
            info_bits_all = u_hat[self.info_indices]
            valid = []
            for l in range(active):
                uh = paths[l].bits[0, :].astype(int)
                uh[self.frozen_bits] = 0
                payload = uh[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((pms[l], uh))
            if valid:
                valid.sort(key=lambda x: x[0])
                u_hat = valid[0][1]

        return u_hat, float(pms[best])
