"""
Offline-friendly shims for TabPFN optional utilities.

Upstream TabPFN depends on an extra package, `tabpfn-common-utils`, which provides
telemetry helpers. In this CSIRO repository we vendor TabPFN for offline use and
**disable telemetry**.

These minimal stubs provide the small API surface TabPFN imports so that
`import tabpfn` works even when `tabpfn-common-utils` is not installed.

NOTE:
This package lives under `src/` (the project's Python package). TabPFN imports it
as a *top-level* package (`import tabpfn_common_utils`). We therefore install a
runtime alias in `sys.modules` (see `src/tabular/tabpfn_patches.py`) before
importing TabPFN.
"""

from __future__ import annotations

__all__ = ["__version__"]

# A synthetic version so debug tooling has something to display.
__version__ = "0.0.0+csiro-shim"


