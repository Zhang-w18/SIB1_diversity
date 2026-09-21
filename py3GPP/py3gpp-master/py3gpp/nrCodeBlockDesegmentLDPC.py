import numpy as np

from py3gpp.nrDLSCHInfo import getCBSInfo
from py3gpp.nrCRCDecode import nrCRCDecode

# def nrCodeBlockDesegmentLDPC(cbs, bgn, blklen):
#     assert len(cbs.shape) == 2
#     blk = None
#     err = False
#     blk = np.zeros(blklen)
#     if cbs.shape[1] == 1:
#         blk = cbs[:blklen, 0]
#         # there is no appended CRC when there is only one segment
#     else:
#         cbsInfo = getCBSInfo(blklen, bgn)
#         idx = 0
#         for i in np.arange(cbsInfo['C']):
#             if i < cbsInfo['C'] - 1:
#                 blk[idx:][:cbsInfo['CBZ']] = cbs[0 : cbsInfo['CBZ'], i]
#                 _, crc = nrCRCDecode(cbs[0 : cbsInfo['CBZ'] + cbsInfo['Lcb'], i], '24B')
#                 idx += cbsInfo['CBZ']
#             else:
#                 blk[idx:][:blklen - idx] = cbs[0 : blklen - idx, i]
#                 _, crc = nrCRCDecode(cbs[0 : cbsInfo['CBZ'] + cbsInfo['Lcb'], i], '24B')
#
#             if crc != 0:
#                 err = True
#                 print(f'nrCodeBlockDesegmentLDPC: crc error in segment {i}')
#
#     return blk, err
def nrCodeBlockDesegmentLDPC(cbs, bgn, blklen):
    # cbs: shape (K, C)
    cbs = np.asarray(cbs, dtype=int)
    assert cbs.ndim == 2
    C = cbs.shape[1]

    # 单块：规范 C=1, Lcb=0，无 24B 追加；直接取前 blklen 位
    if C == 1:
        return cbs[:blklen, 0], False

    # 多块：按规范均分的 CBZ，逐块去 CRC24B 并拼接
    cbsInfo = getCBSInfo(blklen, bgn)
    CBZ = int(cbsInfo['CBZ'])
    Lcb = int(cbsInfo['Lcb'])
    assert Lcb == 24, "For C>1, Lcb should be 24 per spec"

    out = np.zeros(blklen, dtype=int)
    err = False
    s = 0
    for r in range(C):
        # 前 CBZ 位为数据段，后 24 位为 CRC
        data_crc = cbs[0:CBZ + Lcb, r]
        data = data_crc[:CBZ]
        _, crc = nrCRCDecode(data_crc, '24B')   # 返回 (payload, crc_status)
        out[s:s+CBZ] = data
        s += CBZ
        if crc != 0:
            err = True
            print(f'nrCodeBlockDesegmentLDPC: crc error in segment {r}')

    # 截取恰好 blklen 位（通常 s == blklen）
    return out[:blklen], err