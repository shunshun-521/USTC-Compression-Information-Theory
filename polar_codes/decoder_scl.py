"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _frozen_bool


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(8):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, width)
    crc_bits = np.array([(rem >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, width)
    return rem == 0


class _Path:
    __slots__ = ("pm", "u", "active")

    def __init__(self, N):
        self.pm = 0.0
        self.u = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制，L 较小时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.br = bit_reversal_permutation(N)
        self.frozen = _frozen_bool(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length

    def _pm_update(self, pm, llr, u):
        v = 0.0 if llr >= 0 else 1.0
        if u != v:
            pm += abs(llr)
        return pm

    def _decode_path(self, llr, path, phi):
        """对单条路径计算位置 phi 的 LLR（简化：重用 SC 左子树状态不可行，用完整 SC 代价高）"""
        raise NotImplementedError

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [_Path(self.N)]
        for phi in range(self.N):
            new_paths = []
            for p in paths:
                llr_phi = self._compute_llr_for_path(llr, p.u, phi)
                if self.frozen[phi]:
                    u0 = 0
                    pm = self._pm_update(p.pm, llr_phi, u0)
                    p2 = _Path(self.N)
                    p2.u = p.u.copy()
                    p2.u[phi] = u0
                    p2.pm = pm
                    new_paths.append(p2)
                else:
                    for u_cand in (0, 1):
                        p2 = _Path(self.N)
                        p2.u = p.u.copy()
                        p2.u[phi] = u_cand
                        p2.pm = self._pm_update(p.pm, llr_phi, u_cand)
                        new_paths.append(p2)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen)[0]
            passed = [p for p in paths if crc_check(p.u[info_idx], self.crc_length)]
            if passed:
                best = min(passed, key=lambda x: x.pm)
        return best.u.astype(int), best.pm

    def _compute_llr_for_path(self, llr_ch_br, u_partial, phi):
        """计算当前路径在 phi 处的 LLR（块 SC 单步）"""
        N = self.N

        def llr_at(L, offset, depth, target_phi):
            m = len(L)
            if depth == self.n:
                return L[0]
            h = m // 2
            if target_phi < offset + h:
                return llr_at(f_operation(L[:h], L[h:]), offset, depth + 1, target_phi)
            u_left = u_partial[offset : offset + h]
            return llr_at(g_operation(L[:h], L[h:], u_left), offset + h, depth + 1, target_phi)

        L_leaf = llr_at(llr_ch_br, 0, 0, phi)
        return L_leaf
