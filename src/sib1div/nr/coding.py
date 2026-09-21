"""Standards-aligned SIB1 DL-SCH coding and QPSK soft demapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.config import SimulationConfig

from .legacy_adapter import module


def qpsk_modulate(bits: NDArray[np.integer]) -> NDArray[np.complex128]:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size % 2:
        raise ValueError("QPSK requires an even number of bits")
    pairs = bits.reshape(-1, 2)
    return ((1 - 2 * pairs[:, 0]) + 1j * (1 - 2 * pairs[:, 1])) / np.sqrt(2.0)


def qpsk_llr(symbols: NDArray[np.complexfloating], noise_variance: NDArray | float) -> NDArray[np.float64]:
    symbols = np.asarray(symbols)
    variance = np.maximum(np.asarray(noise_variance, dtype=float), 1e-12)
    scale = 2.0 * np.sqrt(2.0) / variance
    return np.column_stack((scale * symbols.real, scale * symbols.imag)).reshape(-1)


@dataclass(frozen=True)
class DecodeResult:
    payload: NDArray[np.uint8]
    crc_ok: bool


class SIB1Codec:
    """TS 38.212 CRC16/BG2 LDPC/rate matching and PDSCH scrambling."""

    def __init__(self, config: SimulationConfig):
        nr = config.data["nr"]
        self.tbs = int(nr["tbs_bits"])
        self.coded_bits = config.coded_bits
        self.rate = float(nr["target_code_rate_x1024"]) / 1024.0
        self.rv = int(nr["rv"])
        self.rnti = int(nr["si_rnti"])
        self.n_id = int(nr.get("scrambling_id", nr.get("n_id", 0)))
        self.info = module("nrDLSCHInfo").nrDLSCHInfo(self.tbs, self.rate)
        if self.info["CRC"] != "16" or self.info["BGN"] != 2:
            raise ValueError("first-round SIB1 must use CRC16 and LDPC base graph 2")

    def scrambling_bits(self) -> NDArray[np.uint8]:
        c_init = (self.rnti << 15) + self.n_id
        return module("nrPRBS").nrPRBS(c_init, self.coded_bits).astype(np.uint8)

    def encode(self, payload: NDArray[np.integer]) -> NDArray[np.uint8]:
        payload = np.asarray(payload, dtype=int).reshape(-1)
        if payload.size != self.tbs or np.any((payload != 0) & (payload != 1)):
            raise ValueError(f"payload must contain exactly {self.tbs} binary bits")
        block = module("nrCRCEncode").nrCRCEncode(payload, self.info["CRC"])
        segments = module("nrCodeBlockSegmentLDPC").nrCodeBlockSegmentLDPC(block, self.info["BGN"])
        encoded = module("nrLDPCEncode").nrLDPCEncode(segments, self.info["BGN"])
        matched = module("nrRateMatchLDPC").nrRateMatchLDPC(
            encoded, self.coded_bits, self.rv, "QPSK", 1
        ).astype(np.uint8)
        return matched ^ self.scrambling_bits()

    def decode(self, scrambled_llr: NDArray[np.floating], iterations: int = 25) -> DecodeResult:
        llr = np.asarray(scrambled_llr, dtype=float).reshape(-1)
        if llr.size != self.coded_bits:
            raise ValueError(f"expected {self.coded_bits} LLR values")
        descrambled = llr * (1 - 2 * self.scrambling_bits().astype(float))
        recovered = module("nrRateRecoverLDPC").nrRateRecoverLDPC(
            descrambled, self.tbs, self.rate, self.rv, "QPSK", 1
        )
        decoded, _ = module("nrLDPCDecode").nrLDPCDecode(
            recovered, self.info["BGN"], iterations, blklen=self.tbs
        )
        block, _ = module("nrCodeBlockDesegmentLDPC").nrCodeBlockDesegmentLDPC(
            decoded, self.info["BGN"], self.tbs + self.info["L"]
        )
        payload, error = module("nrCRCDecode").nrCRCDecode(block, self.info["CRC"])
        return DecodeResult(np.asarray(payload, dtype=np.uint8).reshape(-1), bool(np.asarray(error).item() == 0))

