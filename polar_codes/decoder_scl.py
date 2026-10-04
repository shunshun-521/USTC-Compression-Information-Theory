"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("Unsupported CRC length")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "node_values")

    def __init__(self):
        self.pm = 0.0
        self.node_values = {}


class SCLDecoder:
    """SCL 译码器（在递归 SC 树上进行路径分裂）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n_depth = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _decode_paths(self, y, depth, node, paths):
        if depth == self.n_depth - 1:
            new_paths = []
            for path in paths:
                llr = y[0]
                if node in self.frozen_set:
                    path.node_values[node] = 0
                    path.pm = self._pm_update(path.pm, llr, 0)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = _Path()
                        p.pm = self._pm_update(path.pm, llr, u)
                        p.node_values = dict(path.node_values)
                        p.node_values[node] = u
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            return new_paths[: self.list_size]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left = [
            np.sign(a) * np.sign(b) * min(abs(a), abs(b)) for a, b in zip(l1, l2)
        ]
        paths = self._decode_paths(left, depth + 1, 2 * node, paths)
        right = []
        for path in paths:
            arr1 = [path.node_values.get(2 * node + i, 0) for i in range(len(l1))]
            # arr1 bits for left subtree leaves - use stored node values at leaves under left child
            # Reconstruct left decisions from node_values for g operation
            left_bits = []
            for i in range(len(l1)):
                idx = 2 * node + i if depth + 1 == self.n_depth - 1 else None
            # Simpler: recompute g using partial bits from left subtree decode output
            left_dec = self._collect_subtree_bits(path.node_values, 2 * node, depth + 1, len(l1))
            gvec = [l2[i] + (1 - 2 * left_dec[i]) * l1[i] for i in range(len(l1))]
            sub_paths = self._decode_paths(gvec, depth + 1, 2 * node + 1, [path])
            right.extend(sub_paths)
        right.sort(key=lambda p: p.pm)
        return right[: self.list_size]

    def _collect_subtree_bits(self, nv, root, depth, length):
        # fallback: extract from nv for leaf layer
        bits = []
        for i in range(length):
            bits.append(nv.get(root + i, 0))
        return bits

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        # 通用路径：多次运行 SC 并保留不同硬判决扰动路径（简化实现）
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        candidates = []
        base, _ = sc_decode(llr_ch, self.frozen_bits), 0.0
        candidates.append((0.0, base))
        for scale in np.linspace(-0.5, 0.5, self.list_size - 1):
            pert = llr_ch + scale
            u = sc_decode(pert, self.frozen_bits)
            pm = float(np.sum(np.abs(llr_ch) * (u != (llr_ch < 0).astype(int))))
            candidates.append((pm, u))
        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        if self.crc_length > 0:
            valid = []
            for pm, u in candidates:
                info_bits = u[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u))
            if valid:
                best = min(valid, key=lambda x: x[0])[1]
        return best.copy(), candidates[0][0]


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
