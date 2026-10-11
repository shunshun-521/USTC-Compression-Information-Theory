"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np
from encoder import bit_reversal_permutation


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1 - 2 * x


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma):
    """
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    """
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def codeword_channel_to_decoder(x_ch):
    """信道顺序码字 -> 译码器内部顺序"""
    x_ch = np.asarray(x_ch, dtype=int)
    N = len(x_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return x_ch[inv]


def channel_llr_to_decoder(llr_ch):
    """
    将信道顺序的 LLR 映射为译码器内部顺序（与 polar_encode 的比特倒序配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return llr_ch[inv]
