"""
Created on Thu Nov 11 17:31:11 2025
 Copyright:  Huawei Technologies Co., Ltd. All rights reserved.
 File name: nrPBCHInidicesVariable

 Description:
    use to generate variable length PBCH

 修改记录：
 date name line xxx

@author: m00829866
"""
import numpy as np
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
from py3gpp import nrPRBS, nrSymbolModulate,nrPBCHPRBS,nrCRCEncode,nrRateMatchPolar,nrPolarEncode

#

def nrPBCHDMRS_param(ncellid, ibar_SSB, L_symbols): ## TODO:实现ibarSSB大于8的功能
    assert ncellid >= 0 and ncellid <= 1007
    assert ibar_SSB >= 0 and ibar_SSB <= 7
    if L_symbols <= 0:
        return np.array([], dtype=np.complex128)
    c_init = nrPBCHDMRScinit(ibar_SSB, ncellid)
    c = nrPRBS(c_init, 2*int(L_symbols))

    return nrSymbolModulate(c, 'qpsk')

def nrBCH_param(trblk, sfn, hrf, lssb, idxoffset, ncellid, E = 864):
    # interleaving according to TS38.212 7.1.1
    # fmt: off
    G = [16, 23, 18, 17, 8, 30, 10, 6, 24, 7, 0, 5, 3, 2, 1, 4, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22, 25, 26, 27, 28,
         29, 31]
    # fmt: on
    SFN_PAYLOAD_BEGIN = 1
    SFN_PAYLOAD_LENGTH = 6
    SFN_2ND_LSB = SFN_PAYLOAD_LENGTH + 2
    SFN_3RD_LSB = SFN_PAYLOAD_LENGTH + 1
    #v = 2 * scrblk[G[SFN_3RD_LSB]] + scrblk[G[SFN_2ND_LSB]]
    j_sfn = 0
    j_other = 14
    payload = trblk
    A = 32
    a = np.zeros(A, "int")
    for i in range(24):
        if (i >= SFN_PAYLOAD_BEGIN) and (i < (SFN_PAYLOAD_BEGIN + SFN_PAYLOAD_LENGTH)):
            a[G[j_sfn]] = payload[i]
            j_sfn += 1
        else:
            a[G[j_other]] = payload[i]
            j_other += 1
    a[G[10]] = hrf
    a[G[j_sfn + 0]] = 1 if sfn & 0x08 != 0 else 0
    a[G[j_sfn + 1]] = 1 if sfn & 0x04 != 0 else 0
    a[G[j_sfn + 2]] = 1 if sfn & 0x02 != 0 else 0
    a[G[j_sfn + 3]] = 1 if sfn & 0x01 != 0 else 0

    # scramble
    tmp_seq = nrPBCHPRBS(ncellid, 0, len(a) * 100)
    if lssb in (4, 8):
        M = A - 3
    else:
        M = A - 6
    v = 2 * a[G[SFN_3RD_LSB]] + a[G[SFN_2ND_LSB]]
    n = v * M
    scrambling_seq = tmp_seq[n:][:A]
    scrambling_seq_final = np.zeros(A, "int")
    j = 0
    for i in range(A):
        is_ssb_idx = (i in (G[11], G[12], G[13])) and lssb == 64
        if is_ssb_idx or i == G[10] or i == G[SFN_2ND_LSB] or i == G[SFN_3RD_LSB]:
            scrambling_seq_final[i] = 0
        else:
            scrambling_seq_final[i] = scrambling_seq[j]
            j += 1
    scrblk_scrambled = np.bitwise_xor(scrambling_seq_final, a)

    # attach CRC
    bits = nrCRCEncode(scrblk_scrambled, "24C")[:,0]

    # polar encoding + rate matching
    NMAX = min(int(np.log2(E)),9)
    # NMAX = 9
    encoded = nrPolarEncode(bits, 0, nmax = NMAX, iil = False)
    # rate_matched1 = nrRateMatchPolar(encoded, 0, E, ibil = False)
    rate_matched = nrRateMatchPolar_mh(encoded, A+24, E, I_BIL=False)

    return rate_matched


def nrPBCHDMRScinit(issb, ncellid):
    return 2**11 * (issb + 1) * (ncellid//4 + 1) + 2**6 * (issb + 1) + (ncellid % 4)
def nrPBCHDMRSIndices_param(
    ncellid: int,
    nrb_ssb: int,
    nsym_ssb: int,
    nrb_ss: int,
    style: str = "matlab",
    k_step: int = 4,
    pss_symbol: int = 0,
    sss_symbol: int = 2,
):
    """
    生成参数化的 PBCH DMRS 索引：
      - 频域：每 k_step 个 RE 放 1 个 DMRS（默认 4），频域起点 = (ncellid % k_step)
      - 时域：默认在除 PSS 符号外的所有符号上放 DMRS（即 l != pss_symbol）
      - 在 SSS 符号 (sss_symbol) 上，DMRS 需避开 SSS 的中心频带（宽度 = nrb_ss*12）
    返回：
      style=="matlab": 一维展平索引 (flat = k + l*Nsc)
      style=="python": (k_idx_tuple, l_idx_tuple)
    """
    assert nrb_ssb >= 1 and nsym_ssb >= 1
    Nsc = int(nrb_ssb) * 12
    k0 = ncellid % k_step                     # 频域起点（与旧逻辑兼容）
    ks = np.arange(k0, Nsc, k_step, dtype=int)

    # SSS 中心带（频域）计算
    def center_band_k_indices(Nsc: int, nrb_ss: int):
        if nrb_ss <= 0:
            return np.array([], dtype=int)
        width = min(int(nrb_ss) * 12, Nsc)
        c = Nsc // 2
        k_start = max(c - width // 2, 0)
        k_stop  = min(k_start + width, Nsc)
        return np.arange(k_start, k_stop, dtype=int)

    sss_center_k = center_band_k_indices(Nsc, nrb_ss)

    # 选择放 DMRS 的符号集合：默认所有 l != pss_symbol
    dmrs_symbols = [l for l in range(nsym_ssb) if l != pss_symbol]

    k_all, l_all = [], []
    for l in dmrs_symbols:
        if l == sss_symbol and nsym_ssb > sss_symbol and nrb_ss > 0:
            # SSS 所在符号：只在两侧（去掉中心带）放 DMRS
            ks_use = np.setdiff1d(ks, sss_center_k, assume_unique=False)
        else:
            ks_use = ks
        if ks_use.size == 0:
            continue
        k_all.append(ks_use)
        l_all.append(np.full_like(ks_use, l, dtype=int))

    if not k_all:
        if style == "python":
            return (tuple(), tuple())
        else:
            return np.array([], dtype=int)

    k_cat = np.concatenate(k_all)
    l_cat = np.concatenate(l_all)
    pbch_dmrs_idx =np.asarray(k_cat + l_cat * Nsc, dtype=int)
    if style == "python":
        return (tuple(k_cat.astype(int)), tuple(l_cat.astype(int)))
    elif style == "matlab":
        return pbch_dmrs_idx
    else:
        raise ValueError("Unknown style")

def nrPBCHIndices_param(
    ibar: int,
    nrb_ssb: int,
    nsym_ssb: int,
    nrb_ss: int,
    pss_symbol: int = 0,
    sss_symbol: int = 2,
):
    """
    生成参数化 PBCH 数据 RE 的展平索引 (Matlab 风格: flat = k + l*Nsc)，不包含 DMRS。
    规则：
      - PSS: 固定在符号 l=0 的频域中心带 (nrb_ss 个 RB)，PBCH 不占用该符号。
      - SSS: 固定在符号 l=2 的频域中心带 (nrb_ss 个 RB)，该符号 PBCH 需避开中心带。
      - 其余符号（除 l=0）全部由 PBCH 填满。
    参数：
      ibar: 预留参数（此版本不使用，为兼容后续扩展保留）
      nrb_ssb:  载波 RB 数
      nsym_ssb: 符号数
      nrb_ss: PSS/SSS 占用的 RB 数（中心对称），频域宽度 = nrb_ss*12
    返回：
      ndarray[int]：PBCH 的展平索引（升序，去重）
    """
    assert nrb_ssb >= 1 and nsym_ssb >= 1, "nrb_ssb 与 nsym_ssb 必须为正整数"
    Nsc = int(nrb_ssb) * 12   # 每个符号的子载波数
    ncellid = ibar
    def center_band_flat(Nsc: int, l: int, nrb_ss: int) -> np.ndarray:
        """生成某符号 l 上、以频域中心为对称的连续带（宽度 nrb_ss*12）的展平索引。"""
        if nrb_ss <= 0:
            return np.array([], dtype=int)
        width = int(nrb_ss) * 12
        width = min(width, Nsc)                # 宽度不能超过整列
        c = Nsc // 2
        k0 = c - width // 2
        k1 = k0 + width
        k0 = max(k0, 0); k1 = min(k1, Nsc)     # 边界保护
        if k0 >= k1:
            return np.array([], dtype=int)
        k = np.arange(k0, k1)
        return k + l * Nsc

    # 1) PBCH 不使用符号0；候选符号集合为 [1, 2, ..., nsym_ssb-1]
    pbch_symbols = range(1, nsym_ssb)

    # 2) 收集各符号的 PBCH 索引；在符号2上避开 SSS 中心带
    pbch_list = []
    for l in pbch_symbols:
        sym_all = np.arange(Nsc) + l * Nsc
        if l == 2 and nsym_ssb >= 3 and nrb_ss > 0:
            sss_band = center_band_flat(Nsc, l, nrb_ss)
            sym_pbch = np.setdiff1d(sym_all, sss_band, assume_unique=False)
        else:
            sym_pbch = sym_all
        pbch_list.append(sym_pbch)

    if not pbch_list:
        return np.array([], dtype=int)

    pbch_idx = np.sort(np.concatenate(pbch_list).astype(int))

    dmrs_idx = nrPBCHDMRSIndices_param(ncellid, nrb_ssb, nsym_ssb, nrb_ss, style="matlab",
                                        k_step=4, pss_symbol=pss_symbol, sss_symbol=sss_symbol)
    dmrs_idx = np.asarray(dmrs_idx, dtype=int)
    dmrs_idx = dmrs_idx[(0 <= dmrs_idx) & (dmrs_idx < Nsc * nsym_ssb)]
    pbch_idx = np.setdiff1d(pbch_idx, dmrs_idx, assume_unique=False)

    return pbch_idx


def nrSymbolDemodulate_param(input, mod, nVar=1e-10, csi=1, DecisionType="soft"):
    output = np.empty(0, "float")

    for symbol in input:
        if mod == "BPSK":
            output = np.append(output, np.real(symbol) + np.imag(symbol))
        elif mod == "QPSK":
            output = np.append(output, np.real(symbol))
            output = np.append(output, np.imag(symbol))
        elif mod == "16QAM":
            output = np.append(output, np.real(symbol))
            output = np.append(output, np.imag(symbol))
            output = np.append(output, -(np.abs(np.real(symbol)) - 2 / np.sqrt(10)))
            output = np.append(output, -(np.abs(np.imag(symbol)) - 2 / np.sqrt(10)))

    if DecisionType == "soft":
        output /= nVar / np.exp(1)
        if mod == "16QAM":
            output /= 2
    else:
        output = (output < 0).astype(int)
    output = output * np.repeat(csi, 2)
    return output

def nrSymbolDemodulate_param_persym(input, mod, nVar_cols, re_idx, K, csi=1, DecisionType="soft"):
    input = np.asarray(input)
    re_idx = np.asarray(re_idx, dtype=np.int64)
    nVar_arr = np.asarray(nVar_cols, dtype=float)

    # 兼容两种输入：
    # 1) 每个 symbol 一个 nVar
    # 2) 每个 RE 一个 nVar
    if nVar_arr.size == len(re_idx):
        N0_sym = nVar_arr
    else:
        sym_ids = re_idx // K
        N0_sym = nVar_arr[sym_ids]

    mod_u = mod.upper()

    if mod_u == "BPSK":
        bps = 1
    elif mod_u == "QPSK":
        bps = 2
    elif mod_u == "16QAM":
        bps = 4
    elif mod_u == "64QAM":
        bps = 6
    else:
        raise ValueError(f"Unsupported modulation: {mod}")

    xr = np.real(input)
    xi = np.imag(input)

    if mod_u == "BPSK":
        # 保持你原来的对角 BPSK 风格
        llr_mat = (xr + xi)[:, None]

    elif mod_u == "QPSK":
        llr_mat = np.column_stack([
            xr,   # bit0
            xi    # bit1
        ])

    elif mod_u == "16QAM":
        a = 2 / np.sqrt(10)
        llr_mat = np.column_stack([
            xr,                  # bit0
            xi,                  # bit1
            a - np.abs(xr),      # bit2
            a - np.abs(xi),      # bit3
        ])

    elif mod_u == "64QAM":
        a = 4 / np.sqrt(42)      # inner/outer threshold
        b = 2 / np.sqrt(42)      # inner-pair discrimination level

        llr_mat = np.column_stack([
            xr,                               # bit0
            xi,                               # bit1
            a - np.abs(xr),                   # bit2
            a - np.abs(xi),                   # bit3
            b - np.abs(np.abs(xr) - a),       # bit4
            b - np.abs(np.abs(xi) - a),       # bit5
        ])

    if DecisionType == "soft":
        # 每个符号的噪声方差扩展到每个 bit
        N0_eff = np.repeat(N0_sym, bps)

        output = llr_mat.reshape(-1) / (N0_eff / np.exp(1))

        # 这几项是经验缩放，不是严格 max-log；
        # 保持和你原函数风格一致
        if mod_u == "16QAM":
            output /= 2.0
        elif mod_u == "64QAM":
            output /= 4.0
    else:
        output = (llr_mat.reshape(-1) < 0).astype(int)

    # CSI 也要按每个符号 bps 次扩展
    csi = np.asarray(csi)
    if csi.ndim == 0:
        output = output * csi
    else:
        output = output * np.repeat(csi, bps)

    return output
def nrEqualizeMMSE_persym(rxSym, hest, nVar,pbch_idx,K):
    # 每个 RE 属于哪一列（符号索引 ℓ）
    sym_ids = (pbch_idx // K).astype(np.int64)  # ℓ in [0, N-1]
    # 给每个 RE 取对应列的噪声方差
    nVar_re = nVar[sym_ids]  # (M,)

    # MMSE 等化
    a = np.abs(hest * np.conj(hest))
    csi = a + nVar_re  # 分母
    # 防止极端零分母
    csi = np.where(csi == 0, 1e-30, csi)
    rxEq = rxSym * np.conj(hest) / csi

    return rxEq, csi

# --- §5.4.1.1 Sub-block interleaving ---
def subblock_interleaving_idx(N: int) -> np.ndarray:
    """Return J(n) for sub-block interleaver. N must be multiple of 32."""
    assert N % 32 == 0, "N must be a multiple of 32 per 38.212 §5.4.1.1"
    # Table 5.4.1.1-1: P(i)
    P = np.array(
        [0,1,2,4,3,5,6,7, 8,16,9,17,10,18,11,19,
         12,20,13,21,14,22,15,23,24,25,26,28,27,29,30,31], dtype=int
    )
    n = np.arange(N, dtype=int)
    i = (32 * n) // N
    J = P[i] * (N // 32) + (n % (N // 32))
    return J

def subblock_interleave_bits(d_bits: np.ndarray) -> np.ndarray:
    J = subblock_interleaving_idx(d_bits.size)
    return d_bits[J]

def subblock_deinterleave_vals(vals: np.ndarray) -> np.ndarray:
    """Inverse for LLR/值域：把 y 还原回 d 的顺序，用于译码端恢复 Polar 输入。"""
    N = vals.size
    J = subblock_interleaving_idx(N)
    # invJ = np.empty(N, dtype=int); invJ[J] = np.arange(N)
    out = np.empty_like(vals); out[J] = vals
    return out
# --- §5.4.1.3 Coded-bit interleaver (I_BIL) ---
def coded_bit_interleaver(e: np.ndarray) -> np.ndarray:
    """三角交织：写入上三角(行优先)，按列优先读出。"""
    E = e.size
    T = 0
    while T*(T+1)//2 < E:
        T += 1
    v = np.full((T, T), -1, dtype=int)
    k = 0
    for i in range(T):
        for j in range(T - i):
            if k < E:
                v[i, j] = e[k]; k += 1
    out = np.empty(E, dtype=e.dtype); k = 0
    for j in range(T):
        for i in range(T - j):
            if v[i, j] != -1:
                out[k] = v[i, j]; k += 1
    return out

def coded_bit_deinterleaver(f: np.ndarray) -> np.ndarray:
    """逆三角交织：按列写入上三角，再按行读出。"""
    E = f.size
    T = 0
    while T*(T+1)//2 < E:
        T += 1
    v = np.full((T, T), -1, dtype=int)
    k = 0
    for j in range(T):
        for i in range(T - j):
            if k < E:
                v[i, j] = f[k]; k += 1
    out = np.empty(E, dtype=f.dtype); k = 0
    for i in range(T):
        for j in range(T - i):
            out[k] = v[i, j]; k += 1
    return out
def nrRateMatchPolar_mh(enc_bits: np.ndarray, K: int, E: int, I_BIL: bool=False) -> np.ndarray:
    """
    38.212 §5.4：子块交织 -> 环形缓冲选择(重复/打孔/缩短) -> (可选)coded-bit 交织
    enc_bits: 长度 N (N 为 32 的倍数) 的编码比特 d[]
    K: 信息+CRC 长度（用于决定打孔/缩短阈值）
    E: 输出长度 (<=8192)
    I_BIL: 是否启用末段 coded-bit 交织（PBCH=0）
    """
    N = enc_bits.size
    assert N % 32 == 0 and 1 <= E <= 8192

    # §5.4.1.1
    y = subblock_interleave_bits(enc_bits)
    # §5.4.1.2
    if E >= N:
        # repetition: ek = y[k mod N]
        if E == N:
            e = y.copy()
        else:
            q, r = divmod(E, N)
            e = np.concatenate([np.tile(y, q), y[:r]]).astype(enc_bits.dtype, copy=False)
    else:
        # E < N : puncturing vs shortening
        if K / E <= 7/16:
            # puncturing: ek = y[k + N - E]
            e = y[N - E : N].copy()
        else:
            # shortening: ek = y[k]
            e = y[:E].copy()

    # §5.4.1.3
    if I_BIL:
        e = coded_bit_interleaver(e)

    return e
def nrRateRecoverPolar_mh(llr_f: np.ndarray, K: int, N: int, I_BIL: bool=False,
                       discardRepetition: bool=False, llr_inf: float=1e3) -> np.ndarray:
    """
    译码侧：撤销 I_BIL -> 还原 y 的长度 N 软值 -> 撤销子块交织 -> 送入 Polar 解码器
    llr_f: 长度 E 的 LLR（rm 输出的 f[] 的软值）
    """
    E = llr_f.size
    assert N % 32 == 0

    # 撤销 §5.4.1.3
    if I_BIL:
        llr_e = coded_bit_deinterleaver(llr_f)
    else:
        llr_e = llr_f

    # 还原 §5.4.1.2 的 y[]
    yy = np.copy(llr_e[:N])
    for k in range(N, E):
        yy[k % N] += llr_e[k]
    if E >= N:
        y = np.zeros(N, dtype=llr_e.dtype)
        q, r = divmod(E, N)
        # 先直接累加所有的完整块
        for i in range(q):
            y += llr_e[i * N: (i + 1) * N]  # 按每个块累加
        # 然后处理剩余部分
        if r > 0:
            y[:r] += llr_e[q * N: q * N + r]
    else:
        y = np.empty(N, dtype=llr_e.dtype)
        if K / E <= 7/16:
            # puncturing: y[0:N-E] 未发送 -> LLR=0
            y[:N - E] = 0.0
            y[N - E:] = llr_e
        else:
            # shortening: y[E:N] 未发送，发送端设为 0 -> LLR=+inf
            y[:E] = llr_e
            y[E:] = llr_inf

    # 撤销 §5.4.1.1 （把 y 还原到 Polar 解码需要的 d 顺序）
    # dec_in = subblock_deinterleave_vals(y)
    # d = np.zeros(N)
    # np.put(d, subblock_interleaving(np.arange(N)), y)
    dec_in = subblock_deinterleave_vals(y)
    return dec_in