# ext_scl/scl_adapter.py
import numpy as np
import tensorflow as tf

from sionna.phy.fec.polar.utils import generate_5g_ranking
from sionna.phy.fec.polar.decoding import PolarSCLDecoder
from py3gpp.helper import polar_precode_interleave
from ext_scl2.encoding import PolarEncoder, Polar5GEncoder
from ext_scl2.decoding import PolarSCLDecoder, Polar5GDecoder
_DECODER_CACHE: dict = {}
# 缓存 ranking，避免反复构造大结构

# 对齐Py3gpp版本
def _cached_ranking(K: int, N: int):
    frozen_pos, info_pos = generate_5g_ranking(K, N)
    # 转成 tuple 以便可缓存（不可变）
    return tuple(frozen_pos.tolist()), tuple(info_pos.tolist())

def _get_decoder(K: int, N: int, list_size: int, crc_degree: str, ind_iil_inv:None,
                 use_fast_scl: bool = True, cpu_only: bool = True, return_crc_status: bool=True) -> PolarSCLDecoder:
    key = (K, N, list_size, crc_degree, use_fast_scl, cpu_only)
    dec = _DECODER_CACHE.get(key)
    if dec is None:
        frozen_pos, _ = _cached_ranking(K, N)
        frozen_pos = np.array(frozen_pos, dtype=np.int32)
        ## TODO 新加的
        # enc = Polar5GEncoder(k=K, n=N)
        # dec = Polar5GDecoder(Polar5GEncoder(k=K, n=N), dec_type="SCL", list_size=8,return_crc_status=True,)
        dec = PolarSCLDecoder(frozen_pos,
                              n=N,
                              list_size=list_size,
                              crc_degree=crc_degree,   # 让 Sionna 自己做 CRC 选择
                              use_fast_scl=use_fast_scl,
                              cpu_only=cpu_only,
                              ind_iil_inv = ind_iil_inv,
                              return_crc_status=return_crc_status)  # 要 CRC 状态便于调试
        _DECODER_CACHE[key] = dec
        print("build decoder once")
    return dec
def polar_decode_scl_llr(rec_llr: np.ndarray,
                         K: int,
                         N: int,
                         list_size: int = 8,
                         crc_degree: str = "CRC24C",
                         iil=True,
                         inv_iil: np.ndarray | None = None):
    """
    用 Sionna 的 CA-SCL(L=list_size) 对 Polar 码做译码（CPU 路径）。
    参数:
      rec_llr : 长度 N 的 LLR（nrRateRecoverPolar 的输出）
      K       : 信息比特+CRC 的总长度（PBCH=56）
      N       : 母码长度 (2^nmax)，例如 512 或 1024
      list_size: SCL 列表大小，基线常用 8
      crc_degree: 'CRC24C'（PBCH），或 'CRC11'/'CRC6'（其他控制信道）
      inv_iil : (可选) 输入去交织的逆映射（若你的链路里做了 IIL，则把逆映射传进来）
    返回:
      decoded_bits: shape (K,) 的 0/1 numpy 数组（去掉 CRC 后的信息位需你自己再 nrCRCDecode）
      crc_ok      : bool，若启用 CRC（上面 crc_degree 非 None），就返回 CRC 判定
    """
    assert len(rec_llr) == N, f"rec_llr must have length N={N}"
    # 生成冻结位位置
    # frozen_pos, info_pos = generate_5g_ranking(K, N)

    # 1) 构造逆置换
    if iil:
        perm = polar_precode_interleave(K)
        inv_iil = np.empty(K, dtype=int)
        inv_iil[perm] = np.arange(K)
        ind_iil_inv = tf.constant(inv_iil, dtype=tf.int32)
        # 构建 SCL 解码器（CPU 路径，返回 CRC 判定）  ## TODO: inv_iil Wrong
        dec = _get_decoder(K, N, list_size=list_size, crc_degree='CRC24C', use_fast_scl=True, cpu_only=True,
                           ind_iil_inv=ind_iil_inv, return_crc_status=True)
        # ,ind_iil_inv=ind_iil_inv
        # dec = PolarSCLDecoder(frozen_pos, n=N, list_size=list_size, crc_degree='CRC24C',
        # cpu_only=True, use_fast_scl=True,return_crc_status=True)
    else:
        # 构建 SCL 解码器（CPU 路径，返回 CRC 判定）  ## TODO: inv_iil Wrong
        dec = _get_decoder(K, N, list_size=list_size, crc_degree='CRC24C', use_fast_scl=True, cpu_only=True,ind_iil_inv=None, return_crc_status=True)
        # ,ind_iil_inv=ind_iil_inv
        # dec = PolarSCLDecoder(frozen_pos, n=N, list_size=list_size, crc_degree='CRC24C',
        # cpu_only=True, use_fast_scl=True,return_crc_status=True)

    # Sionna 的 decoder 接口要求输入 tf.Tensor，内部会对符号做 llr = -logit 变换
    x = tf.convert_to_tensor((-rec_llr)[None, :], dtype=tf.float32)  # shape (1, N)

    # 输出是去掉 CRC 后的信息位（若设置了 crc_degree），以及 CRC 状态
    u_hat, crc_status = dec(x)  # u_hat shape: (1, K-CRC)  or (1, K) 取决于 Sionna的定义
    iil = True
    if iil:
        # deinterleave
        p_IL = polar_precode_interleave(K)
        decoded2 = np.empty(u_hat.shape, int)
        np.put(decoded2, p_IL, u_hat)
        u_hat = decoded2.astype(np.int32).reshape(-1)
    else:
        u_hat = u_hat.numpy().astype(np.int32).reshape(-1)
    crc_ok = bool(crc_status.numpy().reshape(()))

    # 如果你需要 **保留 K 比特（含 CRC）** 再走你自己的 nrCRCDecode，
    # 可以把 crc_degree=None 传入构造器，然后这里自己 gather `info_pos`
    # 示例（不启用 Sionna 的 CRC）：见方案 B 的实现提示
    # u_hat = u_hat[inv_iil[:len(u_hat)]]
    return u_hat,crc_ok