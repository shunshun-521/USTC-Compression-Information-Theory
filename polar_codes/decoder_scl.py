"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _build_tree
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _llr_at_phase(root, frozen_bits, u_prefix, phi):
    """在给定前缀 u[0:phi] 下，计算第 phi 个比特的 LLR"""

    def rec(node):
        if node.is_leaf:
            idx = node.lane_id
            if idx < phi:
                node.s[0] = u_prefix[idx]
                return None
            if idx == phi:
                return float(node.lam[0])
            return None

        half = len(node.lam) // 2
        left, right = node.left, node.right
        for i in range(half):
            left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        val = rec(left)
        if val is not None:
            return val
        for i in range(half):
            right.lam[i] = g_operation(node.lam[i], node.lam[half + i], left.s[i])
        val = rec(right)
        if val is not None:
            return val
        for i in range(half):
            node.s[i] = (left.s[i] ^ right.s[i]) % 2
            node.s[half + i] = right.s[i]
        return None

    out = rec(root)
    if out is None:
        raise RuntimeError("failed to compute LLR at phase")
    return out


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径仅保存 u 与 PM，LLR 树共享信道观测）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self._root_template = _build_tree(N)

    def _pm_inc(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N = self.N

        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=np.int8)}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                root = _build_tree(N)
                root.lam[:] = llr
                llr_phi = _llr_at_phase(root, self.frozen_bits, path["u"], phi)

                if self.frozen_bits[phi]:
                    path["pm"] += self._pm_inc(llr_phi, 0)
                    path["u"][phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._pm_inc(llr_phi, bit),
                            "u": path["u"].copy(),
                        }
                        child["u"][phi] = bit
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        best = paths[0]
        if self.crc_length > 0:
            info_bits = best["u"][~self.frozen_bits]
            valid = [p for p in paths if crc_check(p["u"][~self.frozen_bits], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u"].astype(int), float(best["pm"])
