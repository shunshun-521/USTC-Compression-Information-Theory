"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, _propagate_lr


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    """计算 CRC 余数（不附加到消息）"""
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用标准多项式：r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    remainder = _crc_remainder(bits[:-crc_length], crc_length)
    expected = bits[-crc_length:]
    actual = np.array([(remainder >> (crc_length - 1 - i)) & 1
                       for i in range(crc_length)], dtype=int)
    return np.array_equal(expected, actual)


class Path:
    """SCL 译码单条路径"""

    __slots__ = ('L', 'R', 'pm', 'u_hat', 'active')

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1))
        self.R = np.zeros((N, n + 1))
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def _copy_path(self, src):
        """浅拷贝路径"""
        p = Path(self.n, self.N)
        p.L = src.L.copy()
        p.R = src.R.copy()
        p.pm = src.pm
        p.u_hat = src.u_hat.copy()
        return p

    def _setup_r(self, path, phi):
        """根据已译码比特设置 R 消息"""
        path.R[:] = 0.0
        path.R[self.frozen_bits, 0] = self.LARGE
        for i in range(phi):
            if not self.frozen_bits[i]:
                path.R[i, 0] = self.LARGE if path.u_hat[i] == 0 else -self.LARGE

    def _pm_penalty(self, llr_val, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm) 最优路径的估计序列和路径度量
        """
        paths = [Path(self.n, self.N)]
        paths[0].L[:, self.n] = llr_ch.copy()

        for phi in range(self.N):
            new_paths = []

            for path in paths:
                self._setup_r(path, phi)
                _propagate_lr(path.L, path.R, self.n, self.N)
                llr_val = path.L[phi, 0] + path.R[phi, 0]

                if self.frozen_bits[phi]:
                    new_p = self._copy_path(path)
                    new_p.u_hat[phi] = 0
                    new_p.pm += self._pm_penalty(llr_val, 0)
                    new_paths.append(new_p)
                else:
                    for u_bit in (0, 1):
                        new_p = self._copy_path(path)
                        new_p.u_hat[phi] = u_bit
                        new_p.pm += self._pm_penalty(llr_val, u_bit)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths
                        if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
