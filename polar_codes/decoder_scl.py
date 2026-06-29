"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import bit_reversed, f_operation, g_operation, _active_llr_level, _active_bit_level


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07；CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
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

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class Path:
    """SCL 单条路径（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "u_hat", "active")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1))
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        int(top_bit),
                    )

    def _path_update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._path_llr(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr_val, 0)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._path_update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._pm_penalty(llr_val, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            used_parent = {}

            for pm, pidx, bit in candidates:
                if len(new_paths) >= self.list_size:
                    break
                parent = paths[pidx]
                key = (pidx, bit)
                if bit is None:
                    if pidx not in used_parent:
                        parent.pm = pm
                        used_parent[pidx] = True
                        new_paths.append(parent)
                else:
                    if key in used_parent:
                        continue
                    if pidx in used_parent and used_parent.get(pidx) == "full":
                        child = Path(self.N, self.n)
                        child.L[:] = parent.L
                        child.B[:] = parent.B
                        child.u_hat[:] = parent.u_hat
                        child.pm = parent.pm
                    elif pidx not in used_parent:
                        child = parent
                        used_parent[pidx] = "full"
                    else:
                        child = Path(self.N, self.n)
                        child.L[:] = parent.L
                        child.B[:] = parent.B
                        child.u_hat[:] = parent.u_hat
                        child.pm = parent.pm

                    child.pm = pm
                    child.u_hat[l] = bit
                    child.B[l, self.n] = bit
                    self._path_update_bits(child, l)
                    new_paths.append(child)
                    used_parent[key] = True

            paths = new_paths[: self.list_size]
            for p in paths:
                p.active = True

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_idx = np.where(~self.frozen_bits)[0]
                info_bits = path.u_hat[info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


def scl_decode(llr_ch, frozen_bits, list_size=4, crc_length=0):
    """便捷函数接口"""
    decoder = SCLDecoder(len(llr_ch), frozen_bits, list_size, crc_length)
    return decoder.decode(llr_ch)
