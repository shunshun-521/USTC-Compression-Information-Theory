"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import NEG_INF, _compute_llr, _frozen_mask, f_operation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    """按 MSB 优先逐比特更新 CRC 寄存器"""
    mask = (1 << reg_bits) - 1
    top = 1 << (reg_bits - 1)
    crc = 0
    for bit in bits:
        inv = (int(bit) ^ (crc >> (reg_bits - 1))) & 1
        crc = (crc << 1) & mask
        if inv:
            crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).reshape(-1)
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = _crc_remainder(info_bits, poly, reg_bits)
    crc_bits = np.array(
        [(reg >> (reg_bits - 1 - i)) & 1 for i in range(reg_bits)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int).reshape(-1)
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, reg_bits) == 0


def _llr_to_bit(llr):
    return 1 if llr < 0 else 0


def _pm_penalty(llr, u):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    preferred = _llr_to_bit(llr)
    return 0.0 if u == preferred else abs(llr)


class _Path:
    __slots__ = ("llrs", "s", "pm", "u_hat")

    def __init__(self, n, N):
        self.llrs = NEG_INF * np.ones((n + 1, N), dtype=np.float64)
        self.s = -np.ones((n + 1, N), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.llrs.shape[0] - 1, self.llrs.shape[1])
        p.llrs = self.llrs.copy()
        p.s = self.s.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制 + 列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = self.list_size

        paths = [_Path(n, N)]
        paths[0].llrs[n, :] = llr_ch

        for phi in range(N):
            new_paths = []
            for path in paths:
                path.llrs[0 : self.n, :] = NEG_INF
                if self.frozen[phi]:
                    llr_val = _compute_llr(0, phi, path.llrs, path.s)
                    path.pm += _pm_penalty(llr_val, 0)
                    path.s[0, phi] = 0
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    llr_val = _compute_llr(0, phi, path.llrs, path.s)
                    for u in (0, 1):
                        child = path.copy()
                        child.llrs[0 : self.n, :] = NEG_INF
                        child.pm += _pm_penalty(llr_val, u)
                        child.s[0, phi] = u
                        child.u_hat[phi] = u
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L]

        best = self._select_path(paths)
        return best.u_hat.copy(), best.pm

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
