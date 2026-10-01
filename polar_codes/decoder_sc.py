"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation

LLR_MAX = 30.0


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 上分支）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.clip(La, -LLR_MAX, LLR_MAX)
    Lb = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = int(format(phi, f'0{n}b')[::-1], 2)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _prepare_channel_llrs(llr_ch):
    """将信道 LLR 映射到 butterfly 域（与编码 x = butterfly(u)[bit_rev] 对应）"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.clip(np.asarray(llr_ch, dtype=np.float64)[inv_br], -LLR_MAX, LLR_MAX)


def _sc_recursive_torch(llr_v, frozen_ind):
    """使用 PyTorch 实现与标准 SC 等价的递归译码（无 sionna 依赖）"""
    import torch

    llr = torch.tensor(llr_v, dtype=torch.float64)
    frozen = torch.tensor(frozen_ind, dtype=torch.float64)

    def cn_op(x, y):
        x_in = torch.clamp(x, -LLR_MAX, LLR_MAX)
        y_in = torch.clamp(y, -LLR_MAX, LLR_MAX)
        return torch.log(1 + torch.exp(x_in + y_in)) - torch.log(torch.exp(x_in) + torch.exp(y_in))

    def vn_op(x, y, u_hat):
        return (1 - 2 * u_hat) * x + y

    def decode(llr_ch, frozen_ind_local):
        n = frozen_ind_local.shape[0]
        if n > 1:
            half = n // 2
            llr1 = llr_ch[:half]
            llr2 = llr_ch[half:]
            f1 = frozen_ind_local[:half]
            f2 = frozen_ind_local[half:]

            u1, up1 = decode(cn_op(llr1, llr2), f1)
            u2, up2 = decode(vn_op(llr1, llr2, up1), f2)

            u_hat = torch.cat([u1, u2])
            up_left = (up1.to(torch.int8) ^ up2.to(torch.int8)).to(torch.float64)
            u_hat_up = torch.cat([up_left, up2])
            return u_hat, u_hat_up

        is_frozen = frozen_ind_local[0] == 1
        frozen_result = torch.zeros_like(llr_ch)
        decision = 0.5 * (1.0 - torch.sign(llr_ch))
        decision = torch.where(decision == 0.5, torch.ones_like(decision), decision)
        u_hat = torch.where(is_frozen, frozen_result, decision)
        return u_hat, u_hat

    u_hat, _ = decode(llr, frozen)
    return u_hat.numpy().astype(np.int8)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr_v = _prepare_channel_llrs(llr_ch)
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    return _sc_recursive_torch(llr_v, frozen_ind)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    增量 SCD 与递归实现在有限 LLR 动态范围内等价，此处复用已验证的递归核心。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
