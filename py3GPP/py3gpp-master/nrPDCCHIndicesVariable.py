"""
Created on Thu Jun  3 17:31:11 2025
 Copyright:  Huawei Technologies Co., Ltd. All rights reserved.
 File name: nrPDCCHInidicesVariable

 Description:
    use to generate variable length PDCCH

 修改记录：
 date name line xxx

@author: m00829866
"""
import numpy as np
from py3gpp import nrPRBS, nrSymbolModulate,nrPBCHPRBS,nrCRCEncode,nrRateMatchPolar,nrPolarEncode

#
def nrEqualizeMMSE_persym(rxSym, hest, nVar,PDCCH_idx,K):
    # 每个 RE 属于哪一列（符号索引 ℓ）
    sym_ids = (PDCCH_idx // K).astype(np.int64)  # ℓ in [0, N-1]
    # 给每个 RE 取对应列的噪声方差
    nVar_re = nVar[sym_ids]  # (M,)

    # MMSE 等化
    a = np.abs(hest * np.conj(hest))
    csi = a + nVar_re  # 分母
    # 防止极端零分母
    csi = np.where(csi == 0, 1e-30, csi)
    rxEq = rxSym * np.conj(hest) / csi

    return rxEq, csi


def nrPDCCH_param(
    dci_bits: np.ndarray,
    E: int,
    rnti_crc: int,         # CRC 掩码用：SIB1=0xFFFF，UE-specific=C-RNTI
    n_id_scram: int,       # PDCCH 数据扰码用：pdcch-DMRS-ScramblingID 或 NcellID
    n_rnti_scram: int = 0, # PDCCH 数据扰码的 nRNTI；common 场景建议=0
    nmax_cap: int = 9,
) -> np.ndarray:
    a = np.asarray(dci_bits, dtype=np.int8).reshape(-1)

    # 1) CRC + 掩码（返回 [payload||masked_crc]）
    a_crc_masked = _dci_crc24c_with_rnti_mask(a, rnti_crc)

    # 2) Polar 编码 + 速率匹配到 E
    nmax = min(int(np.ceil(np.log2(max(E, 1)))), int(nmax_cap))
    enc = nrPolarEncode(a_crc_masked, 0, nmax=nmax, iil=True)
    # cw_bits_no_scram = nrRateMatchPolar(enc, 0, E, ibil=False).astype(np.int8)
    cw_bits_no_scram = nrRateMatchPolar_mh(enc, len(dci_bits) + 24, E, I_BIL=False).astype(np.int8)

    # 3) PDCCH 数据扰码（注意与 PDSCH 不同）
    c_init = (int(n_rnti_scram) << 16) + int(n_id_scram)% (1 << 31)
    scr = nrPRBS(c_init, E).astype(np.uint8)
    cw_bits = (cw_bits_no_scram ^ scr).astype(np.int8)
    return cw_bits


# =======================================
# 2) CRC24C + RNTI 掩码（38.212 DCI 规则）
# =======================================
def _dci_crc24c_with_rnti_mask(payload_bits: np.ndarray, rnti: int) -> np.ndarray:
    """
    输入 payload，比特向量；对 CRC24C 做 RNTI 掩码后，返回 [payload || masked_crc]。
    兼容两种 nrCRCEncode 返回形式：
      - 仅 24 位 CRC
      - [payload || CRC]（长度 = A + 24）
    """
    a = np.asarray(payload_bits, dtype=np.int8).reshape(-1)

    # 可能返回 24 或 A+24
    crc_out = nrCRCEncode(a, "24C")
    crc_out = np.asarray(crc_out, dtype=np.int8).reshape(-1)

    if crc_out.size == 24:
        crc24 = crc_out
    elif crc_out.size == a.size + 24:
        crc24 = crc_out[-24:]   # 取最后 24 位作为 CRC
    else:
        raise ValueError(
            f"nrCRCEncode returned {crc_out.size} bits; expected 24 or payload+24 (got A={a.size})."
        )

    # 生成 24 位 RNTI 掩码（把 16bit RNTI 循环铺到 24 位）
    rnti_bits16 = np.array([(rnti >> i) & 1 for i in range(16)], dtype=np.int8)
    mask24 = np.concatenate([rnti_bits16, rnti_bits16[:8]]).astype(np.int8)

    masked_crc = (crc24 ^ mask24).astype(np.int8)
    return np.concatenate([a, masked_crc], axis=0)


# ======================================
# 3) PDCCH 扰码 PRBS（38.211 控制信道）
# ======================================
def _pdcch_scrambling_bits(E: int, n_id: int, n_rnti: int = 0) -> np.ndarray:
    """
    生成长度 E 的 PDCCH 扰码比特序列（0/1）：
      c_init = (n_rnti << 16) + n_id
    - Common PDCCH：通常 n_rnti=0，n_id=N_cellID（或高层 pdcch-DMRS-ScramblingID）
    - UE-specific：n_rnti=C-RNTI
    """
    c_init = (int(n_rnti) << 16) + int(n_id)
    return nrPRBS(c_init, E).astype(np.uint8)



# def nrPDCCHDMRScinit(ncellid: int, pdcch_dmrs_scrambling_id: int | None = None) -> int:
#     """
#     返回 DM-RS 序列的 nID。
#     - 若上层给了 pdcch-DMRS-ScramblingID 就用它；
#     - 否则用 NcellID。
#     """
#     if pdcch_dmrs_scrambling_id is not None:
#         return int(pdcch_dmrs_scrambling_id)
#     return int(ncellid)
def nrPDCCHDMRScinit(slot_in_frame: int = 0,
                     ofdm_symbol_in_slot: int = 3,
                     ncellid: int =0,
                     n_sym_per_slot: int = 14) -> int:
    """
    c_init = 2^17 * ((n_sym_per_slot * slot_in_frame + l + 1) * (2*ID + 1) + 2*ID) mod 2^31
    """
    ID = int(ncellid)  # 0..65535（若未显式给出即用 N_ID^cell）
    term = (n_sym_per_slot * int(slot_in_frame) + int(ofdm_symbol_in_slot) + 1)
    c_init = (1 << 17) * (term * (2*ID + 1) + 2*ID)
    return int(c_init % (1 << 31))
def nrPDCCHDMRS_param(ncellid: int, num_re: int, pdcch_dmrs_scrambling_id: int | None = None) -> np.ndarray:
    """
    生成 PDCCH DM-RS 的 QPSK 序列（一次性生成 num_re 个 RE）。
    - ncellid: 0..1007
    - num_re : DMRS RE 总数（注意：不是符号数！）
    - 返回: complex64，QPSK(1/sqrt(2) 归一化)
    """
    assert 0 <= ncellid <= 1007
    if num_re <= 0:
        return np.zeros((0,), dtype=np.complex64)

    n_id = nrPDCCHDMRScinit(ncellid=ncellid)
    # 生成 2*num_re 个扰码比特
    c = nrPRBS(int(n_id), 2 * int(num_re)).astype(np.uint8)

    return nrSymbolModulate(c, 'qpsk')

def nrPDCCHDMRSIndices_param(
        nrb_pdcch: int,
        nsym_pdcch: int,
        comb: int = 4,
        dmrs_on_symbols: tuple | None = None,  # None 表示 nsym_pdcch 个符号全放 DMRS
        style: str = "matlab",
):
    """
    生成 PDCCH DMRS 的索引（频域 comb=4，带小区ID偏移）。
    - ncellid: 小区ID，用于决定频域偏移（k_off = ncellid % comb）
    - nrb_pdcch: PDCCH 频域占用的 RB 数
    - nsym_pdcch: PDCCH 占用的符号数（通常为 CORESET 的 1~3 符号）
    - comb: 频域 comb 间隔（默认 4）
    - dmrs_on_symbols: 指定在哪些 l 上放 DMRS（例如 (0,1,2)）。None 表示 0..nsym_pdcch-1 全部
    - style: "matlab" 返回展平索引 k + l*Nsc；"python" 返回 (k_tuple, l_tuple)
    """
    assert nrb_pdcch >= 1 and nsym_pdcch >= 1
    Nsc = int(nrb_pdcch) * 12
    if dmrs_on_symbols is None:
        dmrs_on_symbols = tuple(range(nsym_pdcch))
    else:
        for l in dmrs_on_symbols:
            assert 0 <= l < nsym_pdcch, "dmrs_on_symbols 中的 l 越界"

    # 频域 comb=4，按小区ID取偏移
    k_off = 1
    ks = np.arange(k_off, Nsc, comb, dtype=int)  # e.g., k = k_off, k_off+4, ...

    if ks.size == 0:
        return np.array([], dtype=int) if style == "matlab" else (tuple(), tuple())

    # 堆叠所有放 DMRS 的符号
    k_list, l_list = [], []
    for l in dmrs_on_symbols:
        k_list.append(ks)
        l_list.append(np.full_like(ks, l, dtype=int))
    k_idx = np.concatenate(k_list) if k_list else np.array([], dtype=int)
    l_idx = np.concatenate(l_list) if l_list else np.array([], dtype=int)

    if style == "python":
        return (tuple(k_idx.tolist()), tuple(l_idx.tolist()))
    elif style == "matlab":
        return (k_idx + l_idx * Nsc).astype(int)
    else:
        raise ValueError("Unknown style")


def nrPDCCHIndices_param(
        nrb_pdcch: int,
        nsym_pdcch: int,
        comb: int = 4,
        dmrs_on_symbols: tuple | None = None,
        style: str = "matlab",
):
    """
    生成 PDCCH 的数据 RE 索引：先生成 DMRS，再从全区域中剔除。
    - 频域 comb=4（可调），DMRS 频域偏移由 ncellid%comb 决定
    - 默认所有 PDCCH 符号上都有 DMRS；可通过 dmrs_on_symbols 指定子集
    """
    assert nrb_pdcch >= 1 and nsym_pdcch >= 1
    Nsc = int(nrb_pdcch) * 12
    all_idx = np.arange(Nsc * nsym_pdcch, dtype=int)

    dmrs_idx = nrPDCCHDMRSIndices_param(
        nrb_pdcch=nrb_pdcch,
        nsym_pdcch=nsym_pdcch,
        comb=comb,
        dmrs_on_symbols=dmrs_on_symbols,
        style="matlab",
    )

    if dmrs_idx.size > 0:
        dmrs_idx = dmrs_idx[(dmrs_idx >= 0) & (dmrs_idx < Nsc * nsym_pdcch)]
        data_idx = np.setdiff1d(all_idx, dmrs_idx, assume_unique=False)
    else:
        data_idx = all_idx

    if style == "matlab":
        return data_idx
    elif style == "python":
        k_idx = data_idx % Nsc
        l_idx = data_idx // Nsc
        return (tuple(k_idx.tolist()), tuple(l_idx.tolist()))
    else:
        raise ValueError("Unknown style")

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
    output = np.empty(0, "float")
    re_idx = np.asarray(re_idx, dtype=np.int64)
    sym_ids = (re_idx // K)  # 每个 RE 的符号列 ℓ
    N0_re = np.asarray(nVar_cols, float)[sym_ids]
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
        nVar_eff = np.repeat(N0_re, 2) if mod.upper() == "QPSK" else N0_re
        output /= nVar_eff / np.exp(1)
        if mod == "16QAM":
            output /= 2
    else:
        output = (output < 0).astype(int)
    output = output * np.repeat(csi, 2)
    return output

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
def subblock_interleaving(u):
    N = u.shape[-1]
    assert np.mod(N, 32) == 0, \
        "length for sub-block interleaving must be a multiple of 32."
    y = np.zeros_like(u)
    # Permutation according to Tab 5.4.1.1-1 in 38.212
    perm = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                     12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27,
                     29, 30, 31])

    for n in range(N):
        i = int(32 * n / N)
        j = int(perm[i] * N / 32 + np.mod(n, N / 32))
        y[n] = u[j]

    return y