"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, precompute_sc_indices, _reorder_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if crc_length == 8:
                msb = (reg >> 7) & 1
                reg = (reg << 1) & 0xFF
            else:
                msb = (reg >> 15) & 1
                reg = (reg << 1) & 0xFFFF
            if msb:
                reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if crc_length == 8:
                msb = (reg >> 7) & 1
                reg = (reg << 1) & 0xFF
            else:
                msb = (reg >> 15) & 1
                reg = (reg << 1) & 0xFFFF
            if msb:
                reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat", "active")

    def __init__(self, n, N):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 P/C）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_at_phi(self, path, phi):
        P, C = path.P, path.C
        for layer in self.llr_layer_vec[phi]:
            psi = phi // self.lambda_offset[layer]
            pm = self.lambda_offset[layer + 1]
            for omega in range(0, self.N, 2 * pm):
                base = omega + (psi // pm) * pm
                left = base
                right = base + pm
                if psi % 2 == 0:
                    P[layer, left] = f_operation(P[layer + 1, left], P[layer + 1, right])
                else:
                    P[layer, right] = g_operation(
                        P[layer + 1, left], P[layer + 1, right], C[layer, left]
                    )
        return P[0, phi]

    def _bit_propagate(self, path, phi):
        P, C = path.P, path.C
        for layer in self.bit_layer_vec[phi]:
            pm = self.lambda_offset[layer + 1]
            if (phi // self.lambda_offset[layer]) % 2 == 0:
                C[layer + 1, phi] = C[layer, phi]
                C[layer + 1, phi + pm] = C[layer, phi]
            else:
                C[layer + 1, phi] = C[layer, phi]
                C[layer + 1, phi - pm] = C[layer, phi - pm] ^ C[layer, phi]

    def _update_llr_layers(self, path, phi):
        """使用与 sc_decode_indexed 相同的更新规则"""
        P, C = path.P, path.C
        n, N = self.n, self.N
        l = 0
        p = phi
        while p & 1:
            p >>= 1
            l += 1
        for layer in range(n - 1, l - 1, -1):
            stride = 1 << (n - 1 - layer)
            block_start = (phi // (2 * stride)) * (2 * stride)
            for j in range(block_start, block_start + stride):
                P[layer, j] = f_operation(P[layer + 1, j], P[layer + 1, j + stride])
                P[layer, j + stride] = g_operation(
                    P[layer + 1, j], P[layer + 1, j + stride], C[layer, j]
                )

    def _bit_back(self, path, phase):
        C = path.C
        n, N = self.n, self.N
        p = phase + 1
        l = 0
        while p & 1:
            p >>= 1
            l += 1
        for layer in range(l):
            stride = 1 << (n - 1 - layer)
            pos = phase % (2 * stride)
            if pos < stride:
                C[layer + 1, phase] = C[layer, phase]
                C[layer + 1, phase + stride] = C[layer, phase]
            else:
                C[layer + 1, phase] = C[layer, phase]
                C[layer + 1, phase - stride] = C[layer, phase - stride] ^ C[layer, phase]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = _reorder_llr(llr_ch)
        n, N = self.n, self.N
        paths = [_Path(n, N)]
        paths[0].P[n, :] = llr_ch

        for phi in range(N):
            candidates = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llr_layers(path, phi)
                llr = path.P[0, phi]

                if self.frozen_bits[phi]:
                    pen = self._pm_penalty(llr, 0)
                    new_path = path
                    new_path.pm += pen
                    new_path.u_hat[phi] = 0
                    new_path.C[0, phi] = 0
                    self._bit_back(new_path, phi)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        cp = _Path(n, N)
                        cp.P = path.P.copy()
                        cp.C = path.C.copy()
                        cp.pm = path.pm + self._pm_penalty(llr, u)
                        cp.u_hat = path.u_hat.copy()
                        cp.u_hat[phi] = u
                        cp.C[0, phi] = u
                        self._bit_back(cp, phi)
                        candidates.append(cp)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat, best.pm
