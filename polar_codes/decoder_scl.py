"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _prepare_channel_llrs, _sc_recursive_torch, LLR_MAX


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    bits = list(info_bits.astype(np.int8))
    for _ in range(crc_length):
        bits.append(0)
    for i in range(len(info_bits)):
        if bits[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    bits[i + j] ^= 1
    return np.array(bits[-crc_length:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _scl_recursive_torch(llr_v, frozen_ind, list_size):
    """基于 PyTorch 的递归 SCL 译码核心"""
    import torch

    llr = torch.tensor(llr_v, dtype=torch.float64)
    frozen = torch.tensor(frozen_ind, dtype=torch.float64)

    def cn_op(x, y):
        x_in = torch.clamp(x, -LLR_MAX, LLR_MAX)
        y_in = torch.clamp(y, -LLR_MAX, LLR_MAX)
        return torch.log(1 + torch.exp(x_in + y_in)) - torch.log(torch.exp(x_in) + torch.exp(y_in))

    def vn_op(x, y, u_hat):
        return (1 - 2 * u_hat) * x + y

    def pm_add(llr_val, bit):
        if abs(llr_val) < 1e-9:
            hard = 1.0
        else:
            hard = 0.0 if llr_val >= 0 else 1.0
        return 0.0 if bit == hard else max(abs(llr_val), 1e-9)

    def decode_list(llr_ch, frozen_local):
        n = frozen_local.shape[0]
        if n == 1:
            paths = []
            if frozen_local[0] == 1:
                paths.append((torch.tensor([0.0]), torch.tensor([0.0]), 0.0))
            else:
                llr0 = llr_ch[0]
                for bit in (0.0, 1.0):
                    paths.append((torch.tensor([bit]), torch.tensor([bit]), pm_add(llr0, bit)))
            paths.sort(key=lambda x: x[2])
            return paths[:list_size]

        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        f1 = frozen_local[:half]
        f2 = frozen_local[half:]

        left_paths = decode_list(cn_op(llr1, llr2), f1)
        all_paths = []
        for u1, up1, pm1 in left_paths:
            llr2_in = vn_op(llr1, llr2, up1)
            right_paths = decode_list(llr2_in, f2)
            for u2, up2, pm2 in right_paths:
                u_hat = torch.cat([u1, u2])
                up_left = (up1.to(torch.int8) ^ up2.to(torch.int8)).to(torch.float64)
                u_hat_up = torch.cat([up_left, up2])
                all_paths.append((u_hat, u_hat_up, pm1 + pm2))

        all_paths.sort(key=lambda x: x[2])
        return all_paths[:list_size]

    paths = decode_list(llr, frozen)
    best_u, _, best_pm = paths[0]
    return best_u.numpy().astype(np.int8), best_pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_ind = self.frozen_bits.astype(np.float64)

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)"""
        if self.list_size == 1:
            u_hat = _sc_recursive_torch(_prepare_channel_llrs(llr_ch), self.frozen_ind)
            return u_hat, 0.0

        llr_v = _prepare_channel_llrs(llr_ch)
        u_hat, pm = _scl_recursive_torch(llr_v, self.frozen_ind, self.list_size)

        if self.crc_length > 0:
            info = u_hat[self.info_indices]
            if crc_check(info, self.crc_length):
                return u_hat, pm
            llr_v2 = llr_v.copy()
            u_hat2, pm2 = _scl_recursive_torch(llr_v2, self.frozen_ind, self.list_size * 2)
            info2 = u_hat2[self.info_indices]
            if crc_check(info2, self.crc_length):
                return u_hat2, pm2

        return u_hat, pm
