import numpy as np

def bit_reversed(x, n):
    """
    Bit-reversal operation.

    Parameters
    ----------
    x: ndarray<int>, int
        a vector of indices
    n: int
        number of bits per index in ``x``

    Returns
    ----------
    ndarray<int>, int
        bit-reversed version of x

    """

    result = 0
    for i in range(n):  # for each bit number
        if (x & (1 << i)):  # if it matches that bit
            result |= (1 << (n - 1 - i))  # set the "opposite" bit in result
    return result

def logdomain_sum(x, y):
    """
    Find the addition of x and y in log-domain. It uses log1p to improve numerical stability.

    Parameters
    ----------
    x: float
        any number in the log-domain
    y: float
        any number in the log-domain

    Returns
    ----------
    float
        the result of x + y

    """

    if x > y:
        z = x + np.log1p(np.exp(y - x))
    else:
        z = y + np.log1p(np.exp(x - y))
    return z

def hard_decision(y):
    """
        Hard decision of a log-likelihood.
    """

    if y >= 0:
        return 0
    else:
        return 1

def upper_llr(l1, l2):
    """
    Update top branch LLR in the log-domain.
    This function supports shortening by checking for infinite LLR cases.

    Parameters
    ----------
    l1: float
        input LLR corresponding to the top branch
    l2: float
        input LLR corresponding to the bottom branch

    Returns
    ----------
    float
        the top branch LLR at the next stage of the decoding tree

    """

    # check for infinite LLR, used in shortening
    if l1 == np.inf and l2 != np.inf:
        return l2
    elif l1 != np.inf and l2 == np.inf:
        return l1
    elif l1 == np.inf and l2 == np.inf:
        return np.inf
    else:  # principal decoding equation
        return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)

def lower_llr(l1, l2, b):
    """
    Update bottom branch LLR in the log-domain.
    This function supports shortening by checking for infinite LLR cases.

    Parameters
    ----------
    l1: float
        input LLR corresponding to the top branch
    l2: float
        input LLR corresponding to the bottom branch
    b: int
        the decoded bit of the top branch

    Returns
    ----------
    float, np.nan
        the bottom branch LLR at the next stage of the decoding tree
    """

    if b == 0:  # check for infinite LLRs, used in shortening
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        else:  # principal decoding equation
            return l1 + l2
    elif b == 1:  # principal decoding equation
        return l1 - l2
    return np.nan

def active_llr_level(i, n):
    """
        Find the first 1 in the binary expansion of i.
    """

    mask = 2**(n-1)
    count = 1
    for k in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

def active_bit_level(i, n):
    """
        Find the first 0 in the binary expansion of i.
    """

    mask = 2**(n-1)
    count = 1
    for k in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

class SCD:
    def __init__(self, myPC):
        self.myPC = myPC
        self.L = np.full((self.myPC.N, self.myPC.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.myPC.N, self.myPC.n + 1), np.nan)
        self.L[:, 0] = self.myPC.likelihoods

    def decode(self):
        """
        Successive Cancellation Decoder. The decoded message is set to ``message_received`` in ``myPC``.
        The decoder will use the frozen set as defined by ``frozen`` in ``myPC``.
        Depends on `update_llrs` and `update_bits`.

        Parameters
        ----------
        y: ndarray<float>
            a vector of likelihoods at the channel output

        -------------
        **References:**

        *  Vangala, H., Viterbo, & Yi Hong. (2014). Permuted successive cancellation decoder for polar codes. 2014 International Symposium on Information Theory and Its Applications, 438–442. IEICE.

        """

        # decode bits in natural order
        for l in [bit_reversed(i, self.myPC.n) for i in range(self.myPC.N)]:
            # evaluate tree of LLRs for root index i
            self.update_llrs(l)

            # make hard decision at output
            if l in self.myPC.frozen:
                self.B[l, self.myPC.n] = 0
            else:
                self.B[l, self.myPC.n] = hard_decision(self.L[l, self.myPC.n])

            # propagate the hard decision just made
            self.update_bits(l)
        return self.B[:, self.myPC.n].astype(int)

    def update_llrs(self, l):
        for s in range(self.myPC.n - active_llr_level(l, self.myPC.n), self.myPC.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.myPC.N, block_size):
                if j % block_size < branch_size:  # upper branch
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:  # lower branch
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(self, l):
        if l < self.myPC.N / 2:
            return

        for s in range(self.myPC.n, self.myPC.n - active_bit_level(l, self.myPC.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:  # lower branch
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]
