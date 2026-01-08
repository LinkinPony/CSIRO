from __future__ import annotations

import importlib
import sys
from typing import Any, Dict


def install_tabpfn_common_utils_shim() -> bool:
    """
    Ensure `import tabpfn_common_utils` works (offline-friendly).

    TabPFN optionally depends on a separate distribution `tabpfn-common-utils`.
    In this repo we provide a minimal no-op implementation under
    `src.tabpfn_common_utils` and alias it into `sys.modules` as a top-level
    package before importing TabPFN.
    """
    try:
        import tabpfn_common_utils  # noqa: F401

        return True
    except Exception:
        pass

    try:
        root = importlib.import_module("src.tabpfn_common_utils")
        sys.modules.setdefault("tabpfn_common_utils", root)

        telemetry = importlib.import_module("src.tabpfn_common_utils.telemetry")
        sys.modules.setdefault("tabpfn_common_utils.telemetry", telemetry)

        interactive = importlib.import_module("src.tabpfn_common_utils.telemetry.interactive")
        sys.modules.setdefault("tabpfn_common_utils.telemetry.interactive", interactive)
        return True
    except Exception:
        return False


def patch_tabpfn_safe_power_force_float64() -> bool:
    """
    Patch TabPFN's SafePowerTransformer to run fit/transform/inverse_transform in float64.

    This mirrors the local-only modifications we avoid keeping in `third_party/TabPFN`
    so that Kaggle/offline runs (read-only third_party) behave the same.
    """
    try:
        import numpy as np
        from tabpfn.preprocessors.safe_power_transformer import SafePowerTransformer
    except Exception:
        return False

    if bool(getattr(SafePowerTransformer, "_csiro_force_float64_patched", False)):
        return True

    orig_fit = SafePowerTransformer.fit
    orig_transform = SafePowerTransformer.transform
    orig_inverse_transform = SafePowerTransformer.inverse_transform

    def fit(self, X: Any, y: Any = None):  # type: ignore[no-untyped-def]
        X64 = np.asarray(X, dtype=np.float64)
        return orig_fit(self, X64, y=y)

    def transform(self, X: Any):  # type: ignore[no-untyped-def]
        X64 = np.asarray(X, dtype=np.float64)
        return orig_transform(self, X64)

    def inverse_transform(self, X: Any):  # type: ignore[no-untyped-def]
        X64 = np.asarray(X, dtype=np.float64)
        return orig_inverse_transform(self, X64)

    SafePowerTransformer.fit = fit  # type: ignore[method-assign]
    SafePowerTransformer.transform = transform  # type: ignore[method-assign]
    SafePowerTransformer.inverse_transform = inverse_transform  # type: ignore[method-assign]
    SafePowerTransformer._csiro_force_float64_patched = True  # type: ignore[attr-defined]
    return True


def patch_tabpfn_transform_borders_one_errstate() -> bool:
    """
    Suppress NumPy `invalid value encountered in greater/less` spam during TabPFN regression.

    TabPFN's regression post-processing compares transformed bar-distribution borders to
    hard limits. When borders contain NaNs (already handled by `~np.isfinite(...)`), NumPy
    may emit RuntimeWarnings on the comparisons. We suppress only `invalid` within the
    `transform_borders_one` call by wrapping it in `np.errstate(invalid='ignore')`.
    """
    try:
        import numpy as np
        import tabpfn.utils as tabpfn_utils
    except Exception:
        return False

    cur = getattr(tabpfn_utils, "transform_borders_one", None)
    if cur is None:
        return False
    if bool(getattr(cur, "_csiro_errstate_invalid_ignore", False)):
        return True

    orig = tabpfn_utils.transform_borders_one

    def wrapped(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        with np.errstate(invalid="ignore"):
            return orig(*args, **kwargs)

    wrapped._csiro_errstate_invalid_ignore = True  # type: ignore[attr-defined]

    # Patch the canonical definition...
    tabpfn_utils.transform_borders_one = wrapped  # type: ignore[assignment]

    # ...and any direct imports (TabPFN regressor imports it into its module namespace).
    try:
        import tabpfn.regressor as tabpfn_regressor

        tabpfn_regressor.transform_borders_one = wrapped  # type: ignore[assignment]
    except Exception:
        pass

    return True


def patch_sklearn_column_transformer_unpickle_compat() -> bool:
    """
    Patch scikit-learn ColumnTransformer to be robust to older pickled state.

    Why:
    - TabPFN fit-states may embed a fitted sklearn `ColumnTransformer` (preprocessor_).
    - When running inference in a different sklearn version (common on Kaggle),
      internal private attributes may be missing after unpickling.
    - Some sklearn versions access `_name_to_fitted_passthrough` during `transform()`.
      If missing, it crashes with:
        AttributeError: 'ColumnTransformer' object has no attribute '_name_to_fitted_passthrough'

    Fix:
    - Wrap `ColumnTransformer.transform` to lazily create the missing attribute.
    - When passthrough entries exist in `transformers_`, populate the mapping with a
      lightweight identity `FunctionTransformer` so transform can proceed safely.
    """
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.preprocessing import FunctionTransformer  # type: ignore
    except Exception:
        return False

    cur = getattr(ColumnTransformer, "transform", None)
    if cur is None:
        return False
    if bool(getattr(cur, "_csiro_unpickle_compat_patched", False)):
        return True

    orig_transform = cur

    def _make_identity_transformer() -> Any:
        # Prefer validate=False (works for numpy arrays and pandas DataFrames).
        try:
            return FunctionTransformer(validate=False)
        except Exception:
            try:
                return FunctionTransformer()
            except Exception:
                return None

    def transform(self, X: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if (not hasattr(self, "_name_to_fitted_passthrough")) or (getattr(self, "_name_to_fitted_passthrough", None) is None):
            mapping: dict[Any, Any] = {}
            try:
                transformers_fitted = getattr(self, "transformers_", None)
                if isinstance(transformers_fitted, (list, tuple)):
                    for t in transformers_fitted:
                        if not (isinstance(t, (list, tuple)) and len(t) >= 2):
                            continue
                        name, trans = t[0], t[1]
                        if trans == "passthrough":
                            ident = _make_identity_transformer()
                            if ident is not None:
                                mapping[name] = ident
            except Exception:
                mapping = {}
            try:
                setattr(self, "_name_to_fitted_passthrough", mapping)
            except Exception:
                pass

        return orig_transform(self, X, *args, **kwargs)

    transform._csiro_unpickle_compat_patched = True  # type: ignore[attr-defined]
    ColumnTransformer.transform = transform  # type: ignore[assignment]
    return True


def apply_tabpfn_runtime_patches() -> Dict[str, bool]:
    """
    Apply all CSIRO runtime patches for TabPFN.

    Returns a dict of patch-name -> applied(bool).
    """
    results: Dict[str, bool] = {}
    results["tabpfn_common_utils_shim"] = install_tabpfn_common_utils_shim()
    results["safe_power_force_float64"] = patch_tabpfn_safe_power_force_float64()
    results["transform_borders_one_errstate"] = patch_tabpfn_transform_borders_one_errstate()
    results["sklearn_column_transformer_unpickle_compat"] = patch_sklearn_column_transformer_unpickle_compat()
    return results


