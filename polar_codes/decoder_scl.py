"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_bp import BPDecoder
from decoder_sc import f_operation


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC_POLYS[crc_length]
    reg = 0
    for b in info_bits:
        reg ^= (b << (crc_length - 1))
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC_POLYS[crc_length]
    reg = 0
    for b in bits:
        reg ^= (b << (crc_length - 1))
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器：在因子图 leaf LLR 上做路径分裂与 PM 剪枝（Lazy Copy 共享信道 LLR）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._bp = BPDecoder(N, self.frozen_bits, max_iter=1, alpha=1.0)

    def _leaf_llrs(self, llr_ch):
        """一次 BP 扫描得到各比特后验 LLR（左端）"""
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6
        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    L[i + k, j - 1] = self._bp._f_min_sum(
                        R[i + k, j] + L[i + k + s, j], L[i + k, j]
                    )
                    L[i + k + s, j - 1] = self._bp._f_min_sum(
                        R[i + k, j], L[i + k, j]
                    ) + L[i + k + s, j]
        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    R[i + k, j + 1] = self._bp._f_min_sum(
                        R[i + k + s, j] + L[i + k + s, j + 1], R[i + k, j]
                    )
                    R[i + k + s, j + 1] = self._bp._f_min_sum(
                        R[i + k, j], L[i + k, j + 1]
                    ) + R[i + k + s, j]
        return L[:, 0] + R[:, 0]

    @staticmethod
    def _pm_add(pm, llr, u):
        v = 0 if llr >= 0 else 1
        return pm + (0.0 if u == v else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        leaf = self._leaf_llrs(llr_ch)

        paths = [{'u': np.zeros(self.N, dtype=np.int8), 'pm': 0.0}]
        for phi in range(self.N):
            new_paths = []
            for st in paths:
                llr = leaf[phi]
                if self.frozen_bits[phi]:
                    new_paths.append({
                        'u': st['u'].copy(),
                        'pm': self._pm_add(st['pm'], llr, 0),
                    })
                    new_paths[-1]['u'][phi] = 0
                else:
                    for u in (0, 1):
                        pu = st['u'].copy()
                        pu[phi] = u
                        new_paths.append({'u': pu, 'pm': self._pm_add(st['pm'], llr, u)})
            new_paths.sort(key=lambda x: x['pm'])
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u'][self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda x: x['pm'])
        else:
            best = min(paths, key=lambda x: x['pm'])

        return best['u'].astype(int), best['pm']
