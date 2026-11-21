"""Compatibility shims so that models pickled with scikit-learn 1.6 still load on newer versions.

The project pins scikit-learn==1.6.1 (the version used for training). If a newer version is installed anyway,
newer ``SimpleImputer.transform`` expects an attribute (``_fill_dtype``) that old pickles do not have; this shim
derives it from the fitted statistics instead of crashing.
"""

from __future__ import annotations


def apply() -> None:
    try:
        from sklearn.impute import SimpleImputer
    except Exception:  # noqa: BLE001
        return
    if "_fill_dtype" in SimpleImputer.__dict__:
        return

    def _get(self):
        if "_fill_dtype_value" in self.__dict__:
            return self.__dict__["_fill_dtype_value"]
        if "_fit_dtype" in self.__dict__:
            return self.__dict__["_fit_dtype"]
        return self.statistics_.dtype

    def _set(self, value):
        self.__dict__["_fill_dtype_value"] = value

    SimpleImputer._fill_dtype = property(_get, _set)


apply()
