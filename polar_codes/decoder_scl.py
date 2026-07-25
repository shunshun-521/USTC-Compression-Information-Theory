"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _LLR_UNSET, _compute_llr


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
CRC16_POLY = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_remainder(msg_bits, generator):
    """对消息比特做 CRC 长除法，返回余数"""
    n = len(generator) - 1
    data = [int(b) for b in msg_bits] + [0] * n
    for i in range(len(msg_bits)):
        if data[i]:
            for j in range(len(generator)):
                data[i + j] ^= generator[j]
    return np.array(data[len(msg_bits) :], dtype=int)


def _crc_verify(bits, generator):
    """验证含 CRC 的比特序列是否合法"""
    data = [int(b) for b in bits]
    n = len(generator) - 1
    for i in range(len(bits) - n):
        if data[i]:
            for j in range(len(generator)):
                data[i + j] ^= generator[j]
    return all(v == 0 for v in data[-n:])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    generator = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_remainder(info_bits, generator)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    generator = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_verify(bits, generator)


class _PathState:
    __slots__ = ("llrs", "bits", "pm", "u_hat")

    def __init__(self, n, N, llr_ch):
        self.llrs = np.full((n + 1, N), np.nan, dtype=np.float64)
        self.llrs[n, :] = llr_ch.copy()
        self.bits = np.full((n + 1, N), -1, dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = _PathState(self.n, self.N, src.llrs[self.n])
        dst.llrs = src.llrs.copy()
        dst.bits = src.bits.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.n, self.N, llr_ch)]

        for phi in range(self.N):
            for path in paths:
                path.llrs[: self.n, :] = _LLR_UNSET
                path.llrs[self.n, :] = llr_ch
                path.bits[:] = -1
                if phi > 0:
                    path.bits[0, :phi] = path.u_hat[:phi]

            candidates = []
            for path in paths:
                llr_phi = _compute_llr(0, phi, path.llrs, path.bits, self.n)

                if self.frozen_bits[phi]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr_phi, 0)
                    new_path.u_hat[phi] = 0
                    new_path.bits[0, phi] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr_phi, bit)
                        new_path.u_hat[phi] = bit
                        new_path.bits[0, phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
