"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _upper_llr,
    channel_llr_to_decoder,
)
from utils import crc_encode as utils_crc_encode
from utils import crc_check as utils_crc_check


def crc_encode(info_bits, crc_length=8):
    return utils_crc_encode(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return utils_crc_check(bits, crc_length)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径间共享 LLR/比特数组，仅在写入时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "active": True}

    def _clone_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "active": True,
        }

    def _path_llr_at_root(self, path):
        return path["L"][0, self.n] if self.n > 0 else path["L"][0, 0]

    def _update_path_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_path_bits(self, path, l):
        _update_bits(path["B"], l, self.n, self.N)

    def _pm_penalty(self, llr, u):
        """与 LLR 符号不一致时加 |LLR|"""
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch, apply_br_reorder=True):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if apply_br_reorder:
            llr_work = channel_llr_to_decoder(llr_ch, self.N)
        else:
            llr_work = llr_ch.copy()

        paths = [self._new_path(llr_work)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                if not path["active"]:
                    continue
                self._update_path_llrs(path, l)
                llr_root = path["L"][l, self.n]

                if l in self.frozen_set:
                    path["pm"] += self._pm_penalty(llr_root, 0)
                    path["B"][l, self.n] = 0
                    self._update_path_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = self._clone_path(path)
                        p["pm"] += self._pm_penalty(llr_root, u)
                        p["B"][l, self.n] = u
                        self._update_path_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]
            for p in paths:
                p["active"] = True

        # 选路径
        best = None
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for p in paths:
                u = p["B"][:, self.n].astype(int)
                info_bits = u[info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best is None or p["pm"] < best["pm"]:
                        best = p
        if best is None:
            best = min(paths, key=lambda p: p["pm"])

        u_hat = best["B"][:, self.n].astype(int)
        return u_hat, best["pm"]
