"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode, _channel_llr_to_decode_order, f_operation, g_operation


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（分层 LLR/比特数组，路径度量 PM）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def _init_path(self, llr_ch):
        L = np.zeros((self.n + 1, self.N), dtype=np.float64)
        B = np.zeros((self.n + 1, self.N), dtype=np.int8)
        L[self.n, :] = llr_ch
        return {"L": L, "B": B, "u": np.zeros(self.N, dtype=np.int8), "pm": 0.0}

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = _channel_llr_to_decode_order(llr_ch)
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            layer = 0
            tmp = phi
            while tmp & 1:
                layer += 1
                tmp >>= 1

            new_paths = []
            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                for l in range(layer, self.n):
                    istep = 1 << (self.n - l)
                    hstep = istep >> 1
                    for i in range(0, self.N, istep):
                        for j in range(i, i + hstep):
                            L[l, j] = f_operation(L[l + 1, j], L[l + 1, j + hstep])
                for l in range(self.n - 1, layer - 1, -1):
                    istep = 1 << (self.n - l)
                    hstep = istep >> 1
                    for i in range(0, self.N, istep):
                        for j in range(i, i + hstep):
                            L[l, j + hstep] = g_operation(
                                L[l + 1, j], L[l + 1, j + hstep], B[l, j]
                            )
                llr_bit = L[layer, phi]
                branches = [(0,)] if self.frozen_bits[phi] else [(0,), (1,)]
                for u_bit in branches:
                    pm_new = pm + (abs(llr_bit) if (u_bit[0] == 0 and llr_bit < 0) or (u_bit[0] == 1 and llr_bit >= 0) else 0.0)
                    if self.frozen_bits[phi]:
                        u_bit = (0,)
                        pm_new = pm + (abs(llr_bit) if llr_bit < 0 else 0.0)
                    Lc = L.copy()
                    Bc = B.copy()
                    Bc[layer, phi] = u_bit[0]
                    uc = path["u"].copy()
                    uc[phi] = u_bit[0]
                    new_paths.append({"L": Lc, "B": Bc, "u": uc, "pm": pm_new})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

            for path in paths:
                l = layer
                while l < self.n:
                    istep = 1 << (self.n - l)
                    hstep = istep >> 1
                    B = path["B"]
                    for i in range(0, self.N, istep):
                        for j in range(i, i + hstep):
                            B[l + 1, j] = B[l, j]
                            B[l + 1, j + hstep] = (B[l, j] + B[l, j + hstep]) % 2
                    l += 1

        candidates = []
        for path in paths:
            u_hat = path["u"].copy()
            u_hat[self.frozen_bits] = 0
            candidates.append((path["pm"], u_hat))

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [(pm, u) for pm, u in candidates if crc_check(u[info_idx], self.crc_length)]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
