"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    x = np.asarray(x)
    return 1.0 - 2.0 * x


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
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * y / (sigma ** 2)


def channel_llr_to_decode_domain(llr_ch):
    """极化译码域 LLR：与 Arikan F^{⊗n} 蝶形顺序对齐（相对信道码字做比特倒序逆映射）"""
    N = len(llr_ch)
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    eb_n0_lin = 10 ** (eb_n0_db / 10.0)
    snr = 2.0 * rate * eb_n0_lin
    return 1.0 / np.sqrt(snr)

