from __future__ import annotations
import io, pathlib, numpy as np, csv
from collections import OrderedDict

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-12

def assert_no_nans(name, arr):
    assert np.isfinite(arr).all(), f"{name}: NaN/Inf"

def assert_nonnegative(name, arr):
    assert (arr >= 0).all(), f"{name}: negative values"

def assert_monotonic_increasing(name, arr):
    assert np.all(np.diff(arr) > 0), f"{name}: must be strictly increasing"

def assert_in_unit_interval(name, arr):
    assert ((arr >= 0) & (arr <= 1)).all(), f"{name}: must be in [0,1]"

def _try_float(s):
    try:
        return float(s)
    except Exception:
        return s


def load_expected_csv(path: str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    out = {}
    for n in data.dtype.names:
        arr = np.asarray(data[n])
        # if string-like, attempt numeric cast
        if arr.dtype.kind in ("U", "S", "O"):
            try:
                arr = arr.astype(float)
            except Exception:
                pass
        out[n] = arr
    return out
def write_diff_csv(tmp_dir: pathlib.Path, label: str, got: dict, ref: dict) -> pathlib.Path:
    keys = sorted(set(got) & set(ref))
    out = tmp_dir / f"diff_{label}.csv"
    cols, header = [], []
    for k in keys:
        g = np.asarray(got[k]).squeeze()
        r = np.asarray(ref[k]).squeeze()
        err = g - r
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.abs(r) > 0, np.abs(err)/np.abs(r), np.nan)
        header += [f"{k}_ref", f"{k}_got", f"{k}_abs_err", f"{k}_rel_err"]
        cols += [r, g, np.abs(err), rel]
    maxlen = max(len(np.ravel(c)) for c in cols) if cols else 0
    table = np.full((maxlen, len(cols)), np.nan) if maxlen else np.zeros((1,1))
    for j, c in enumerate(cols):
        c = np.ravel(c)
        table[: len(c), j] = c
    buf = io.StringIO()
    if header:
        buf.write(",".join(header) + "\n")
        for i in range(maxlen):
            buf.write(",".join("" if (table[i, j] != table[i, j]) else f"{table[i, j]:.8e}" for j in range(table.shape[1])) + "\n")
    out.write_text(buf.getvalue())
    return out
