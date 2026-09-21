import numpy as np
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices

def nrPBCHIndices_param(ibar, nrb: int, nsym: int, nssrb: int) -> np.ndarray:
    """
    生成参数化 PBCH 数据 RE 的展平索引 (Matlab 风格: flat = k + l*Nsc)，不包含 DMRS。
    规则：
      - PSS: 固定在符号 l=0 的频域中心带 (nssrb 个 RB)，PBCH 不占用该符号。
      - SSS: 固定在符号 l=2 的频域中心带 (nssrb 个 RB)，该符号 PBCH 需避开中心带。
      - 其余符号（除 l=0）全部由 PBCH 填满。
    参数：
      ibar: 预留参数（此版本不使用，为兼容后续扩展保留）
      nrb:  载波 RB 数
      nsym: 符号数
      nssrb: PSS/SSS 占用的 RB 数（中心对称），频域宽度 = nssrb*12
    返回：
      ndarray[int]：PBCH 的展平索引（升序，去重）
    """
    assert nrb >= 1 and nsym >= 1, "nrb 与 nsym 必须为正整数"
    Nsc = int(nrb) * 12   # 每个符号的子载波数

    def center_band_flat(Nsc: int, l: int, nssrb: int) -> np.ndarray:
        """生成某符号 l 上、以频域中心为对称的连续带（宽度 nssrb*12）的展平索引。"""
        if nssrb <= 0:
            return np.array([], dtype=int)
        width = int(nssrb) * 12
        width = min(width, Nsc)                # 宽度不能超过整列
        c = Nsc // 2
        k0 = c - width // 2
        k1 = k0 + width
        k0 = max(k0, 0); k1 = min(k1, Nsc)     # 边界保护
        if k0 >= k1:
            return np.array([], dtype=int)
        k = np.arange(k0, k1)
        return k + l * Nsc

    # 1) PBCH 不使用符号0；候选符号集合为 [1, 2, ..., nsym-1]
    pbch_symbols = range(1, nsym)

    # 2) 收集各符号的 PBCH 索引；在符号2上避开 SSS 中心带
    pbch_list = []
    for l in pbch_symbols:
        sym_all = np.arange(Nsc) + l * Nsc
        if l == 2 and nsym >= 3 and nssrb > 0:
            sss_band = center_band_flat(Nsc, l, nssrb)
            sym_pbch = np.setdiff1d(sym_all, sss_band, assume_unique=False)
        else:
            sym_pbch = sym_all
        pbch_list.append(sym_pbch)

    if not pbch_list:
        return np.array([], dtype=int)

    pbch_idx = np.sort(np.concatenate(pbch_list).astype(int))
    return pbch_idx

def nrPBCHIndices(ibar):
    indices = np.hstack((np.arange(240, 480), np.arange(480, 528), np.arange(672, 720), np.arange(720, 960)))

    return np.setdiff1d(indices, nrPBCHDMRSIndices(ibar))
