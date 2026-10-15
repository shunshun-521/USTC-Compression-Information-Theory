"""
极化码 BP 译码器
阻尼迭代消息传递（在蝶形域对 SC 输出做 LLR 反馈），含早停。
"""
import numpy as np

from decoder_sc import _PermutedSCD, _channel_llr_for_decoder, sc_decode
from encoder import bit_reversal_permutation, polar_encode


def _sc_decode_butterfly(llr_bf, frozen_bits, N):
    dec = _PermutedSCD(N, frozen_bits, llr_bf)
    return dec.decode()


class BPDecoder:
    """阻尼迭代 BP 近似。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._inv = np.argsort(bit_reversal_permutation(N))

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_int = _channel_llr_for_decoder(llr_orig, self.N)
        llr_work = llr_int.copy()
        u_hat = np.zeros(self.N, dtype=int)
        num_iters = 0
        scale = max(np.mean(np.abs(llr_int)), 1.0)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            u_hat = _sc_decode_butterfly(llr_work, self.frozen_bits, self.N)
            x_hat = polar_encode(u_hat)
            sign_bf = (1.0 - 2.0 * x_hat)[self._inv]
            llr_work = llr_int + self.alpha * sign_bf * scale

            hard = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        return u_hat, num_iters
