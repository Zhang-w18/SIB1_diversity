"""SIB1 PDSCH DMRS and data RE construction for the fixed first-round slot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.config import SimulationConfig

from .coding import qpsk_modulate
from .legacy_adapter import module


@dataclass(frozen=True)
class ResourceGrid:
    symbols: NDArray[np.complex128]
    data_mask: NDArray[np.bool_]
    dmrs_mask: NDArray[np.bool_]
    dmrs_symbols: tuple[int, ...]

    @property
    def data_coordinates(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return np.nonzero(self.data_mask)


def dmrs_sequence(n_id: int, slot: int, symbol: int, count: int) -> NDArray[np.complex128]:
    c_init = (2**17 * (14 * slot + symbol + 1) * (2 * n_id + 1) + 2 * n_id) % 2**31
    bits = module("nrPRBS").nrPRBS(c_init, 2 * count).astype(np.uint8)
    return qpsk_modulate(bits)


def map_pdsch(config: SimulationConfig, data_symbols: NDArray[np.complexfloating], slot: int = 0) -> ResourceGrid:
    nr = config.data["nr"]
    n_sc = config.active_subcarriers
    n_sym = int(nr["slot_symbols"])
    start = int(nr["pdsch_start_symbol"])
    stop = start + int(nr["pdsch_length_symbols"])
    dmrs_syms = tuple(int(x) for x in nr["dmrs_symbols"])
    grid = np.zeros((n_sym, n_sc), dtype=np.complex128)
    allocated = np.zeros_like(grid, dtype=bool)
    allocated[start:stop, :] = True
    dmrs_mask = np.zeros_like(grid, dtype=bool)
    dmrs_k = np.arange(0, n_sc, 2)
    n_id = int(nr.get("scrambling_id", nr.get("n_id", 0)))
    for symbol in dmrs_syms:
        dmrs_mask[symbol, dmrs_k] = True
        grid[symbol, dmrs_k] = dmrs_sequence(n_id, slot, symbol, dmrs_k.size)
    data_mask = allocated & ~dmrs_mask
    values = np.asarray(data_symbols, dtype=np.complex128).reshape(-1)
    if values.size != int(np.count_nonzero(data_mask)):
        raise ValueError(f"expected {np.count_nonzero(data_mask)} PDSCH data symbols")
    grid[data_mask] = values
    return ResourceGrid(grid, data_mask, dmrs_mask, dmrs_syms)

