"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode, channel_llr_to_decoder, f_operation, g_operation
from utils import crc_encode, crc_check


class SCLDecoder:
    """SCL 译码器：在 Arikan 树上维护至多 L 条路径。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        if self.list_size <= 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr = channel_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        paths = [{"u": np.zeros(self.N, dtype=int), "pm": 0.0}]

        for phi in range(self.N):
            new_paths = []
            for st in paths:
                llr_phi = self._bit_llr(llr, st["u"], phi)
                if self.frozen_bits[phi]:
                    u = st["u"].copy()
                    u[phi] = 0
                    pm = self._pm_update(st["pm"], llr_phi, 0)
                    new_paths.append({"u": u, "pm": pm})
                else:
                    for bit in (0, 1):
                        u = st["u"].copy()
                        u[phi] = bit
                        pm = self._pm_update(st["pm"], llr_phi, bit)
                        new_paths.append({"u": u, "pm": pm})
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for st in paths:
                bits = st["u"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(st)
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        return best["u"], best["pm"]

    def _bit_llr(self, llr, u_prefix, phi):
        """用冻结未来比特的 SC 判决近似当前 LLR。"""
        frozen = self.frozen_bits.copy()
        frozen[:phi] = True
        frozen[phi] = False
        u_tmp = sc_decode(llr, frozen)
        return 10.0 if u_tmp[phi] == 0 else -10.0
