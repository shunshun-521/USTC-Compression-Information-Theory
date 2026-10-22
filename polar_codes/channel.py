"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np

from encoder import bit_reversal_permutation


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    x = np.asarray(x, dtype=int)
    return 1.0 - 2.0 * x


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    s = np.asarray(s, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    """
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    """
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def reorder_llr_for_decoder(llr_ch):
    """
    将信道输出顺序的 LLR 映射为 Vangala/SCD 因子图自然顺序（F^⊗n 域）。
    发送码字 x = u·G_N 含比特倒序 B_N，故 llr_nat[j] = llr_ch[inv_B(j)]。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv_br = np.empty(N, dtype=int)
    inv_br[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]
