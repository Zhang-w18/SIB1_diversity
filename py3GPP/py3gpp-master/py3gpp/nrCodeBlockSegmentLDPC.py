import numpy as np

from py3gpp.nrDLSCHInfo import getCBSInfo
from py3gpp.nrCRCEncode import nrCRCEncode

# def nrCodeBlockSegmentLDPC(blk, bgn):
#     assert bgn in [1, 2], 'bgn must be in [1, 2]'
#     cbsInfo = getCBSInfo(len(blk), bgn)
#     assert cbsInfo['Lcb'] in [0, 24], f'Error: Lcb = {cbsInfo["Lcb"]} is not supported!'
#     if len(blk.shape) == 1:
#         blk = np.expand_dims(blk, axis= 1 )
#     cbs = np.ones((cbsInfo['K'], cbsInfo['C']), dtype=int) * (-1) # fill bits are -1
#     idx = 0
#     for i in np.arange(cbsInfo['C']):
#         if i < cbsInfo['C'] - 1:
#             if cbsInfo['Lcb'] == 0:
#                 cbs[0:cbsInfo['CBZ'], i] = blk[idx:][:cbsInfo['CBZ']]
#             elif cbsInfo['Lcb'] == 24:
#                 cbs[0:cbsInfo['CBZ'] + cbsInfo['Lcb'], i] = nrCRCEncode(blk[idx:, 0][:cbsInfo['CBZ']], '24B', 0)[:, 0]
#             idx += cbsInfo['CBZ']
#         else:
#             cbs[0: cbsInfo['CBZ'], i] = 0 ## Null?
#             if cbsInfo['Lcb'] == 0:
#                 cbs[0: len(blk) - idx, i] = blk[idx:, 0]
#             elif cbsInfo['Lcb'] == 24:
#                 blk_padded = np.zeros(cbsInfo['CBZ'])
#                 blk_padded[: len(blk) - idx] = blk[idx:, 0]
#                 cbs[0: cbsInfo['CBZ'] + cbsInfo['Lcb'], i] = nrCRCEncode(blk_padded, '24B', 0)[:, 0]
#
#     return cbs

def nrCodeBlockSegmentLDPC(blk, bgn):
    assert bgn in [1, 2], 'bgn must be in [1, 2]'
    cbsInfo = getCBSInfo(len(blk), bgn)
    # 期望: cbsInfo 提供 C, K, Kp(=K'), CBZ(=K'-Lcb), Lcb∈{0,24}
    assert cbsInfo['Lcb'] in [0, 24], f'Error: Lcb = {cbsInfo["Lcb"]} is not supported!'

    # 统一形状 & 类型
    blk = np.asarray(blk).ravel()               # (B,)
    blk = blk[:, None]                          # (B,1)

    C   = int(cbsInfo['C'])
    K   = int(cbsInfo['K'])                     # 每块最终长度
    Lcb = int(cbsInfo['Lcb'])                   # 0 或 24
    CBZ = int(cbsInfo['CBZ'])                   # 每块“数据段”长度 = K' - Lcb

    # 初始化为 NULL filler (-1)
    cbs = np.full((K, C), fill_value=-1, dtype=int)

    s = 0  # 输入比特读指针
    for r in range(C):
        # 1) 固定长度的数据段：恰为 CBZ，比特从 blk 按顺序取
        data_seg = blk[s:s+CBZ, 0]
        if data_seg.size != CBZ:
            # 正常规范下不应发生；若发生说明 getCBSInfo 未按规范均分
            raise ValueError(f"Data segment size {data_seg.size} != CBZ {CBZ} at block {r}")
        s += CBZ

        if Lcb == 0:
            # 无 CRC：直接放入 [0:CBZ)
            cbs[0:CBZ, r] = data_seg
        else:
            # 有 CRC：对 data_seg 计算 24B CRC，并把 data+CRC 共 CBZ+24 位放入块首
            seg_crc = nrCRCEncode(data_seg, '24B', 0)[:, 0]  # 长度=CBZ+24
            cbs[0:CBZ + Lcb, r] = seg_crc

        # 其余 [K' .. K-1] 的 filler 位置保持 -1（已初始化）

    return cbs