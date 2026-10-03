"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_register(bits, crc_length):
    """CRC-8/16 按字节移位实现（多项式 0x07 / 0x8005）"""
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_register(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), crc_length
    )
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "C", "u_hat", "copy_L", "copy_C")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        self.C = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.copy_L = [True] * (n + 1)
        self.copy_C = [True] * (n + 1)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.lambda_offset = [1 << i for i in range(self.n + 1)]

    def _ensure_owned(self, path, layer, kind="L"):
        flags = path.copy_L if kind == "L" else path.copy_C
        arr_list = path.L if kind == "L" else path.C
        if not flags[layer]:
            arr_list[layer] = arr_list[layer].copy()
            flags[layer] = True

    def _clone_path(self, path):
        child = _Path(self.n, self.N)
        child.pm = path.pm
        child.L = path.L
        child.C = path.C
        child.copy_L = path.copy_L[:]
        child.copy_C = path.copy_C[:]
        child.u_hat = path.u_hat.copy()
        for i in range(self.n + 1):
            child.copy_L[i] = False
            child.copy_C[i] = False
        return child

    def _update_llr(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            self._ensure_owned(path, layer, "L")
            self._ensure_owned(path, layer + 1, "L")
            step = self.lambda_offset[layer]
            L = path.L
            C = path.C
            for block in range(0, self.N, 2 * step):
                for j in range(step):
                    a = block + j
                    b = block + j + step
                    L[layer][a] = f_operation(L[layer + 1][a], L[layer + 1][b])
                    L[layer][b] = g_operation(L[layer + 1][a], L[layer + 1][b], C[layer][a])

    def _bit_propagate(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            self._ensure_owned(path, layer, "C")
            self._ensure_owned(path, layer + 1, "C")
            step = self.lambda_offset[layer]
            C = path.C
            for block in range(0, self.N, 2 * step):
                for j in range(step):
                    a = block + j
                    b = block + j + step
                    C[layer + 1][b] = (C[layer][a] ^ C[layer][b]) & 1
                    C[layer + 1][a] = C[layer][a]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0
        paths = []
        root = _Path(self.n, self.N)
        root.L[self.n][:] = llr_ch
        paths.append(root)

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path.L[0][phi]
                if self.frozen_bits[phi]:
                    new = self._clone_path(path)
                    u = 0
                    new.pm += self._pm_penalty(llr, u)
                    new.u_hat[phi] = u
                    new.C[0][phi] = u
                    self._bit_propagate(new, phi)
                    candidates.append(new)
                else:
                    for u in (0, 1):
                        new = self._clone_path(path)
                        new.pm += self._pm_penalty(llr, u)
                        new.u_hat[phi] = u
                        new.C[0][phi] = u
                        self._bit_propagate(new, phi)
                        candidates.append(new)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best = paths[0]
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p
            if best_crc is not None:
                best = best_crc
        return best.u_hat.copy(), best.pm
