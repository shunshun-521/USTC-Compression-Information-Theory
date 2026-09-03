"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _f_scalar, g_operation, _permute_llr, _to_frozen_set, sc_decode


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYS[crc_length]
    info = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _sub_decode(llr, depth, n, base, decided):
    """在已知 decided[base:...] 时译码子树，返回该子树叶子比特"""
    length = len(llr)
    if depth == n - 1:
        idx = base
        if idx in decided:
            return [decided[idx]]
        return [1 if llr[0] < 0 else 0]

    half = length // 2
    l1, l2 = llr[:half], llr[half:]
    f_out = [float(_f_scalar(a, b)) for a, b in zip(l1, l2)]
    arr1 = _sub_decode(f_out, depth + 1, n, base, decided)
    g_out = [float(g_operation(a, b, u)) for a, b, u in zip(l1, l2, arr1)]
    arr2 = _sub_decode(g_out, depth + 1, n, base + half, decided)
    merged = [(a + b) % 2 for a, b in zip(arr1, arr2)]
    merged.extend(arr2)
    return merged


def _llr_at_bit(llr, depth, n, base, target, decided):
    """计算 target 比特 LLR，decided 包含 <target 的已知比特"""
    length = len(llr)
    if depth == n - 1:
        return float(llr[0])

    half = length // 2
    l1, l2 = llr[:half], llr[half:]
    if target < base + half:
        f_out = [float(_f_scalar(a, b)) for a, b in zip(l1, l2)]
        return _llr_at_bit(f_out, depth + 1, n, base, target, decided)

    f_out = [float(_f_scalar(a, b)) for a, b in zip(l1, l2)]
    arr1 = _sub_decode(f_out, depth + 1, n, base, decided)
    g_out = [float(g_operation(a, b, u)) for a, b, u in zip(l1, l2, arr1)]
    return _llr_at_bit(g_out, depth + 1, n, base + half, target, decided)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = ~self.frozen_bits

    def _penalty(self, llr, u):
        hard = 1 if llr < 0 else 0
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = _permute_llr(llr_ch).tolist()
        paths = [{"pm": 0.0, "decided": {}}]

        for bit_idx in range(self.N):
            new_paths = []
            for path in paths:
                llr_leaf = _llr_at_bit(llr, 0, self.n, 0, bit_idx, path["decided"])

                if bit_idx in self.frozen_set:
                    decided = dict(path["decided"])
                    decided[bit_idx] = 0
                    pm = path["pm"] + self._penalty(llr_leaf, 0)
                    new_paths.append({"pm": pm, "decided": decided})
                else:
                    for u in (0, 1):
                        decided = dict(path["decided"])
                        decided[bit_idx] = u
                        pm = path["pm"] + self._penalty(llr_leaf, u)
                        new_paths.append({"pm": pm, "decided": decided})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_ok = [p for p in paths if self._crc_ok(p["decided"])]
        best = min(crc_ok or paths, key=lambda p: p["pm"])

        u_hat = np.zeros(self.N, dtype=int)
        for i, b in best["decided"].items():
            u_hat[i] = b
        return u_hat, best["pm"]

    def _crc_ok(self, decided):
        if self.crc_length == 0:
            return True
        u_hat = np.zeros(self.N, dtype=int)
        for i, b in decided.items():
            u_hat[i] = b
        return crc_check(u_hat[self.info_mask], self.crc_length)
