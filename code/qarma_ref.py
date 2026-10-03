"""Classical reference implementation of QARMA-64 (Avanzi, IACR ToSC 2017(1), pp. 4-44).

Written directly from the specification.  The published test vectors are only used to
CHECK this implementation (see TV_CIPHER / check_test_vectors); nothing here is fitted to them.
"""

SBOX = {
    0: [0, 14, 2, 10, 9, 15, 8, 11, 6, 4, 3, 7, 13, 12, 1, 5],
    1: [10, 13, 14, 6, 15, 7, 3, 5, 9, 8, 0, 12, 11, 1, 2, 4],
    2: [11, 6, 8, 15, 12, 0, 9, 14, 3, 7, 4, 5, 13, 2, 1, 10],
}
SBOX[3] = [SBOX[2].index(i) for i in range(16)]          # sigma_2^{-1}

TAU = [0, 11, 6, 13, 10, 1, 12, 7, 5, 14, 3, 8, 15, 4, 9, 2]
TAU_INV = [TAU.index(i) for i in range(16)]
H_PERM = [6, 5, 14, 15, 0, 1, 2, 3, 7, 12, 13, 4, 8, 9, 10, 11]
H_INV = [H_PERM.index(i) for i in range(16)]
LFSR_CELLS = [0, 1, 3, 4, 8, 11, 13]
RC = [0x0000000000000000, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89,
      0x452821E638D01377, 0xBE5466CF34E90C6C, 0x3F84D5B5B5470917, 0x9216D5D98979FB1B]
ALPHA = 0xC0AC29B7C97C50DD
MASK64 = (1 << 64) - 1
# MixColumns M = Q = circ(0, rho, rho^2, rho); entry (i,j) is rho^MQ[i][j], zero on the diagonal
MQ = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]


def cells(x):
    return [(x >> (60 - 4 * i)) & 15 for i in range(16)]


def uncells(c):
    v = 0
    for i in range(16):
        v = (v << 4) | c[i]
    return v


def shuffle(x, p):
    c = cells(x)
    return uncells([c[p[i]] for i in range(16)])


def rotl4(v, e):
    return ((v << e) | (v >> (4 - e))) & 15 if e else v


def mixcolumns(x):
    c = cells(x)
    o = [0] * 16
    for col in range(4):
        for row in range(4):
            a = 0
            for k in range(4):
                if row != k:
                    a ^= rotl4(c[4 * k + col], MQ[row][k])
            o[4 * row + col] = a
    return uncells(o)


def subcells(x, s):
    return uncells([s[v] for v in cells(x)])


def lfsr(v):
    return (((v ^ (v >> 1)) & 1) << 3) | (v >> 1)


def lfsr_inv(v):
    return ((v << 1) & 15) | (((v >> 3) ^ v) & 1)


def tweak_fwd(t):
    c = cells(shuffle(t, H_PERM))
    for i in LFSR_CELLS:
        c[i] = lfsr(c[i])
    return uncells(c)


def tweak_bwd(t):
    c = cells(t)
    for i in LFSR_CELLS:
        c[i] = lfsr_inv(c[i])
    return shuffle(uncells(c), H_INV)


def ortho(w):
    """o(w) = (w >>> 1) xor (w >> 63)."""
    return ((w >> 1) | ((w << 63) & MASK64)) ^ (w >> 63)


def encrypt(P, T, w0, k0, r=7, sb=1):
    s = SBOX[sb]
    si = [s.index(i) for i in range(16)]
    w1, k1 = ortho(w0), k0
    x = P ^ w0
    for i in range(r):                                   # forward rounds
        x ^= k0 ^ T ^ RC[i]
        if i:
            x = mixcolumns(shuffle(x, TAU))
        x = subcells(x, s)
        T = tweak_fwd(T)
    x ^= w1 ^ T                                          # central forward round
    x = subcells(mixcolumns(shuffle(x, TAU)), s)
    x = shuffle(mixcolumns(shuffle(x, TAU)) ^ k1, TAU_INV)   # pseudo-reflector
    x = shuffle(mixcolumns(subcells(x, si)), TAU_INV)        # central backward round
    x ^= w0 ^ T
    for i in reversed(range(r)):                         # backward rounds
        T = tweak_bwd(T)
        x = subcells(x, si)
        if i:
            x = shuffle(mixcolumns(x), TAU_INV)
        x ^= k0 ^ T ^ RC[i] ^ ALPHA
    return x ^ w1


# Published test vectors (Avanzi, ToSC 2017(1), Appendix "Test Vectors").
TV = dict(P=0xfb623599da6e8127, T=0x477d469dec0b8762, w0=0x84be85ce9804e94b, k0=0xec2802d4e0a488e9)
TV_CIPHER = {
    (0, 5): 0x3ee99a6c82af0c38, (0, 6): 0x9f5c41ec525603c9, (0, 7): 0xbcaf6c89de930765,
    (1, 5): 0x544b0ab95bda7c3a, (1, 6): 0xa512dd1e4e3ec582, (1, 7): 0xedf67ff370a483f2,
    (2, 5): 0xc003b93999b33765, (2, 6): 0x270a787275c48d10, (2, 7): 0x5c06a7501b63b2fd,
}


def check_test_vectors():
    bad = []
    for (sb, r), c in TV_CIPHER.items():
        got = encrypt(TV['P'], TV['T'], TV['w0'], TV['k0'], r, sb)
        if got != c:
            bad.append((sb, r, hex(got), hex(c)))
    return bad


def self_test():
    """Structural self-tests that do not use the test vectors."""
    assert all(lfsr_inv(lfsr(v)) == v for v in range(16))
    assert all(SBOX[2][SBOX[3][v]] == v for v in range(16))
    assert all(SBOX[0][SBOX[0][v]] == v for v in range(16)), 'sigma_0 must be an involution'
    assert all(SBOX[1][SBOX[1][v]] == v for v in range(16)), 'sigma_1 must be an involution'
    x = 0x0123456789abcdef
    assert mixcolumns(mixcolumns(x)) == x, 'M must be involutory'
    assert tweak_bwd(tweak_fwd(x)) == x
    assert shuffle(shuffle(x, TAU), TAU_INV) == x
    return True


if __name__ == '__main__':
    print('self test:', self_test())
    bad = check_test_vectors()
    print('test vectors:', 'ALL PASS' if not bad else bad)
