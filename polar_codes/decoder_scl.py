"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int)
    n = len(bits)
    nbytes = (n + 7) // 8
    arr = np.zeros(nbytes * 8, dtype=np.uint8)
    arr[:n] = bits
    out = bytearray()
    for i in range(0, len(arr), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | int(arr[i + j])
        out.append(b)
    return bytes(out)


def _crc8_byte(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc8_byte(_bits_to_bytes(info_bits)) if crc_length == 8 else _crc16_byte(_bits_to_bytes(info_bits))
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return _crc8_byte(_bits_to_bytes(bits)) == 0
    return _crc16_byte(_bits_to_bytes(bits)) == 0


def _pm_penalty(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm + (abs(llr) if u != hard else 0.0)


class _SCLPath:
    __slots__ = ("pm", "u_hat", "u_up")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.u_up = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（递归树，Sionna 风格）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = [_SCLPath(self.N)]
        self._decode_recursive(paths, llr_ch, 0, self.N)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _decode_recursive(self, paths, llr_sub, offset, length):
        if length == 1:
            idx = offset
            new_paths = []
            for p in paths:
                llr = llr_sub[0]
                if self.frozen_bits[idx]:
                    if llr < 0:
                        p.pm += abs(llr)
                    p.u_hat[idx] = 0
                    p.u_up[idx] = 0
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        cp = self._copy_path(p)
                        cp.pm = _pm_penalty(cp.pm, llr, u)
                        cp.u_hat[idx] = u
                        cp.u_up[idx] = u
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths[:] = new_paths[: self.list_size]
            return

        half = length // 2
        llr1 = llr_sub[:half]
        llr2 = llr_sub[half:]
        left_off = offset
        right_off = offset + half

        # 左子树
        left_llr = f_operation(llr1, llr2)
        left_paths = [self._copy_path(p) for p in paths]
        self._decode_recursive(left_paths, left_llr, left_off, half)

        # 右子树（每条左路径独立展开，再统一裁剪）
        all_right_paths = []
        for p in left_paths:
            right_llr = g_operation(llr1, llr2, p.u_up[left_off:right_off])
            sub_paths = [self._copy_path(p)]
            self._decode_recursive(sub_paths, right_llr, right_off, half)
            all_right_paths.extend(sub_paths)

        all_right_paths.sort(key=lambda p: p.pm)
        merged = all_right_paths[: self.list_size]

        for p in merged:
            p.u_up[left_off:right_off] = (
                p.u_up[left_off:right_off] ^ p.u_up[right_off:right_off + half]
            ).astype(int)

        paths[:] = merged

    def _copy_path(self, p):
        cp = _SCLPath(self.N)
        cp.pm = p.pm
        cp.u_hat = p.u_hat.copy()
        cp.u_up = p.u_up.copy()
        return cp
