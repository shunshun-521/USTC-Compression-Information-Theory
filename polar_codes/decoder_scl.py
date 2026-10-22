"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive

# CRC-8: 0x07, CRC-16: 0x8005
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    mask = (1 << crc_len) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        if reg & (1 << (crc_len - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("pm", "P", "C", "u_hat", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _advance_path(self, path, phi):
        """对单条路径计算当前根 LLR（与 SC 相同递推）。"""
        n = self.n
        N = self.N
        P = path.P
        C = path.C

        for layer in range(n):
            if ((phi >> layer) & 1) != 0:
                break
            step = 1 << layer
            for block in range(0, N, step << 1):
                left = block
                right = block + step
                P[layer][left] = f_operation(P[layer + 1][left], P[layer + 1][right])

        for layer in range(n - 1, -1, -1):
            if ((phi >> layer) & 1) == 0:
                continue
            step = 1 << layer
            block = (phi >> (layer + 1)) << (layer + 1)
            left = block
            right = block + step
            u_bit = C[layer][left]
            P[layer][right] = g_operation(
                P[layer + 1][left], P[layer + 1][right], u_bit
            )

        return float(P[0][phi])

    def _propagate_bit(self, path, phi, u_bit):
        n = self.n
        N = self.N
        C = path.C
        path.u_hat[phi] = u_bit
        C[0][0] = u_bit
        for layer in range(n):
            if ((phi >> layer) & 1) != 0:
                break
            step = 1 << layer
            block = (phi >> (layer + 1)) << (layer + 1)
            left = block
            right = block + step
            C[layer + 1][left] = C[layer][left] ^ C[layer][right]
            C[layer + 1][right] = C[layer][right]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        n = self.n
        N = self.N
        L = self.list_size

        paths = [_Path(n, N) for _ in range(L)]
        paths[0].P[n, :] = llr_ch
        active_count = 1

        for phi in range(N):
            candidates = []
            for pidx in range(active_count):
                path = paths[pidx]
                llr_root = self._advance_path(path, phi)
                if self.frozen_bits[phi]:
                    u = 0
                    pm = path.pm
                    if llr_root < 0:
                        pm += abs(llr_root)
                    new_path = _Path(n, N)
                    new_path.pm = pm
                    new_path.P = path.P.copy()
                    new_path.C = path.C.copy()
                    new_path.u_hat = path.u_hat.copy()
                    self._propagate_bit(new_path, phi, u)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        pm = path.pm
                        hard = 0 if llr_root >= 0 else 1
                        if u != hard:
                            pm += abs(llr_root)
                        new_path = _Path(n, N)
                        new_path.pm = pm
                        new_path.P = path.P.copy()
                        new_path.C = path.C.copy()
                        new_path.u_hat = path.u_hat.copy()
                        self._propagate_bit(new_path, phi, u)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active_count = min(L, len(candidates))
            paths[:active_count] = candidates[:active_count]

        best = min(paths[:active_count], key=lambda p: p.pm)

        if self.crc_length > 0:
            crc_ok = [
                p
                for p in paths[:active_count]
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
