"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from encoder import channel_llr_to_decoder


def f_operation(La, Lb):
    """Check-node (boxplus) 运算"""
    llr_max = 30.0
    La = np.clip(La, -llr_max, llr_max)
    Lb = np.clip(Lb, -llr_max, llr_max)
    return np.log1p(np.exp(La + Lb)) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """VN 更新"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 Sionna PolarSCDecoder 一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)

    def polar_decode_sc(llr_ch, fz):
        n = len(llr_ch)
        if n > 1:
            half = n // 2
            llr1 = llr_ch[:half]
            llr2 = llr_ch[half:]
            fz1 = fz[:half]
            fz2 = fz[half:]

            x_llr1 = f_operation(llr1, llr2)
            u_hat1, u_hat1_up = polar_decode_sc(x_llr1, fz1)

            x_llr2 = g_operation(llr1, llr2, u_hat1_up)
            u_hat2, u_hat2_up = polar_decode_sc(x_llr2, fz2)

            u_hat = np.concatenate([u_hat1, u_hat2])
            u_hat1_up_i = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(np.float64)
            u_hat_up = np.concatenate([u_hat1_up_i, u_hat2_up])
            return u_hat, u_hat_up

        is_frozen = fz[0] == 1
        if is_frozen:
            u_hat = np.array([0.0])
        else:
            decision = 0.5 * (1.0 - np.sign(llr_ch[0]))
            if decision == 0.5:
                decision = 1.0
            u_hat = np.array([decision])
        return u_hat, u_hat

    u_hat, _ = polar_decode_sc(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（接口占位）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [
        [layer for layer in range(n) if ((phi >> layer) & 1) == 0] for phi in range(N)
    ]
    bit_layer_vec = [
        [layer for layer in range(n) if ((phi >> layer) & 1) == 1] for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归实现暂委托递归版本（保证数值一致）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码入口（LLR 约定：与 channel.compute_llr 一致）"""
    llr_dec = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=np.float64)
    return sc_decode_recursive(llr_dec, frozen_bits)
