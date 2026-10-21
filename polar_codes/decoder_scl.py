"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _is_frozen


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length, poly):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加在信息比特之后（MSB 先）"""
    poly = _crc_poly(crc_length)
    bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_remainder(bits, crc_length, poly)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验尾部 CRC"""
    poly = _crc_poly(crc_length)
    return _crc_remainder(bits, crc_length, poly) == 0


class _Path:
    __slots__ = ("pm", "decisions", "partial_stack")

    def __init__(self, N):
        self.pm = 0.0
        self.decisions = np.zeros(N, dtype=int)
        self.partial_stack = []


class SCLDecoder:
    """SCL 译码器（路径复制；L 较小时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(
            np.asarray(frozen_bits, dtype=int) == 0
        )[0]

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = self._compute_llr(phi, llr, path)
                if _is_frozen(self.frozen_bits, phi):
                    pen = self._llr_penalty(llr_phi, 0)
                    p2 = self._fork(path, phi, 0, pen)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        pen = self._llr_penalty(llr_phi, bit)
                        new_paths.append(self._fork(path, phi, bit, pen))

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L]

        best = self._select_path(paths)
        return best.decisions.copy(), best.pm

    def _fork(self, path, phi, bit, penalty):
        p = _Path(self.N)
        p.pm = path.pm + penalty
        p.decisions = path.decisions.copy()
        p.decisions[phi] = bit
        p.partial_stack = [s.copy() for s in path.partial_stack]
        self._push_partial(p, phi, bit)
        return p

    def _compute_llr(self, phi, channel_llr, path):
        """对单条路径计算当前比特 LLR（递归 SC 树，使用已判决比特）"""
        return self._sc_llr_at(
            channel_llr, list(range(self.N)), path.decisions, phi
        )

    def _sc_llr_at(self, llrs, indices, decisions, target_idx):
        if len(indices) == 1:
            return llrs[0]

        half = len(indices) // 2
        left_idx = indices[:half]
        right_idx = indices[half:]

        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = self._sc_partial(left_llr, left_idx, decisions)

        if target_idx in left_idx:
            pos = left_idx.index(target_idx)
            return self._sc_llr_at(left_llr, left_idx, decisions, target_idx)

        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        return self._sc_llr_at(right_llr, right_idx, decisions, target_idx)

    def _sc_partial(self, llrs, indices, decisions):
        """左子树部分码字（与 SC 递归返回一致）"""
        if len(indices) == 1:
            i = indices[0]
            return [decisions[i]]

        half = len(indices) // 2
        left_idx = indices[:half]
        right_idx = indices[half:]

        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = self._sc_partial(left_llr, left_idx, decisions)

        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = self._sc_partial(right_llr, right_idx, decisions)

        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    def _push_partial(self, path, phi, bit):
        path.decisions[phi] = bit

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p.decisions[self.info_indices]
                if len(info) < self.crc_length:
                    continue
                payload = info[: -self.crc_length]
                crc_part = info[-self.crc_length :]
                if crc_check(np.concatenate([payload, crc_part]), self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
