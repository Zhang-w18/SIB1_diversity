"""Controlled access to the tested legacy NR coding helpers.

The legacy repository is never added to ``sys.path``. Only the explicitly
listed CRC/LDPC/rate-matching/sequence modules are loaded through a private
package namespace. This is temporary until these functions are internalized.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


_PACKAGE = "_sib1div_legacy_py3gpp"
_ALLOWED = {
    "nrDLSCHInfo", "nrCRCEncode", "nrCRCDecode",
    "nrCodeBlockSegmentLDPC", "nrCodeBlockDesegmentLDPC",
    "nrLDPCEncode", "nrLDPCDecode", "nrRateMatchLDPC",
    "nrRateRecoverLDPC", "nrPRBS", "helper", "codes",
}


def _legacy_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "py3GPP" / "py3gpp-master" / "py3gpp"
    if not (root / "nrLDPCEncode.py").exists():
        raise RuntimeError(f"tested legacy NR helpers not found at {root}")
    return root


def module(name: str):
    if name not in _ALLOWED:
        raise ValueError(f"legacy module is not allow-listed: {name}")
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_legacy_root())]
        package.__package__ = _PACKAGE
        sys.modules[_PACKAGE] = package
        # The selected files use absolute ``py3gpp`` imports internally.
        # Alias only the package object; no filesystem search path is changed.
        sys.modules.setdefault("py3gpp", package)
    loaded = importlib.import_module(f"{_PACKAGE}.{name}")
    # Internal absolute imports may load the same source under ``py3gpp.*``.
    return loaded

