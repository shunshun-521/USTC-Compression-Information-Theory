"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
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


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "P", "C", "u_hat", "parent")

    def __init__(self, n, N, llr_ch, parent=None):
        self.pm = 0.0
        if parent is None:
            self.P = np.zeros((n + 1, N), dtype=np.float64)
            self.C = np.zeros((n + 1, N), dtype=np.int8)
            self.P[n, :] = llr_ch
            self.u_hat = np.zeros(N, dtype=int)
            self.parent = None
        else:
            self.P = parent.P
            self.C = parent.C
            self.u_hat = parent.u_hat.copy()
            self.parent = parent


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 u_hat 与必要时深拷贝 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, self.bit_layer_vec = precompute_sc_indices(N)

    def _penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llr_layers(self, P, C, phi, layer):
        n = self.n
        N = self.N
        for l in range(n - 1, layer - 1, -1):
            stride = 1 << l
            num_blocks = N // (2 * stride)
            pl = P[l]
            pl1 = P[l + 1]
            cl = C[l]
            for block in range(num_blocks):
                base = block * 2 * stride
                for j in range(stride):
                    idx = base + j
                    pl[idx] = f_operation(pl1[idx], pl1[idx + stride])
                    pl[idx + stride] = g_operation(
                        pl1[idx], pl1[idx + stride], cl[idx]
                    )

    def _propagate_bits(self, C, phi, layer):
        N = self.N
        for l in range(layer):
            stride = 1 << l
            num_blocks = N // (2 * stride)
            cl = C[l]
            cl1 = C[l + 1]
            for block in range(num_blocks):
                base = block * 2 * stride
                for j in range(stride):
                    idx = base + j
                    cl1[idx] = cl[idx] ^ cl[idx + stride]
                    cl1[idx + stride] = cl[idx + stride]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        paths = [_Path(n, N, llr_ch)]

        for phi in range(N):
            p = phi
            layer = 0
            while p % 2 == 1:
                layer += 1
                p >>= 1

            candidates = []
            for path in paths:
                self._update_llr_layers(path.P, path.C, phi, layer)
                llr0 = path.P[0, phi]

                if self.frozen_bits[phi]:
                    bit = 0
                    new_pm = path.pm + self._penalty(llr0, bit)
                    candidates.append((new_pm, path, bit))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._penalty(llr0, bit)
                        candidates.append((new_pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.list_size]:
                child = _Path(n, N, llr_ch, parent=parent)
                child.pm = pm
                if child.P is parent.P:
                    child.P = parent.P.copy()
                    child.C = parent.C.copy()
                child.u_hat[phi] = bit
                child.C[0, phi] = bit
                self._propagate_bits(child.C, phi, layer)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]
        return best.u_hat.copy(), best.pm
