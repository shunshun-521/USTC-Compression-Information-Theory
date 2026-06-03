"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import f_operation, g_operation, _b_check, _s_updater, _compute_llr


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    """GF(2) CRC 除法，返回余数"""
    reg = [0] * crc_len
    poly_bits = [(poly >> i) & 1 for i in range(crc_len - 1, -1, -1)]
    for bit in data_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if poly_bits[i]:
                    reg[i] ^= fb
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_divide(info_bits.tolist(), poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).flatten()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_divide(bits.tolist(), poly, crc_length)
    return np.all(rem == 0)


class _Path:
    """单条 SCL 路径（Lazy Copy：共享 llrs/s 直至分裂时复制）"""

    __slots__ = ("llrs", "s", "pm", "u_hat", "active")

    def __init__(self, n, N):
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool).flatten()
        self.info_mask = ~self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        p = _Path(self.n, self.N)
        p.llrs[self.n, :] = llr_ch
        return p

    def _copy_path(self, src):
        p = _Path(self.n, self.N)
        p.llrs = src.llrs.copy()
        p.s = src.s.copy()
        p.pm = src.pm
        p.u_hat = src.u_hat.copy()
        return p

    def _path_llr_at(self, path, bit_idx):
        return _compute_llr(0, bit_idx, path.llrs, path.s)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).flatten()
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                llr_val = self._path_llr_at(path, phi)

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_p = self._copy_path(path)
                    new_p.pm += penalty
                    new_p.u_hat[phi] = 0
                    new_p.s[0, phi] = 0
                    new_p.llrs[0, phi] = np.inf
                    candidates.append(new_p)
                else:
                    for bit in (0, 1):
                        new_p = self._copy_path(path)
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (
                            bit == 1 and llr_val < 0
                        ) else abs(llr_val)
                        new_p.pm += penalty
                        new_p.u_hat[phi] = bit
                        new_p.s[0, phi] = bit
                        new_p.llrs[0, phi] = llr_val
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = np.inf
        best_path = paths[0]

        if self.crc_length > 0:
            info_idx = np.where(self.info_mask)[0]
            for p in paths:
                bits = p.u_hat[info_idx]
                if crc_check(bits, self.crc_length) and p.pm < best_pm:
                    best_pm = p.pm
                    best_crc = p
            if best_crc is not None:
                best_path = best_crc
        else:
            best_path = min(paths, key=lambda p: p.pm)

        return best_path.u_hat.copy(), float(best_path.pm)
