"""Cache IO and table helpers shared by the Stage-2 analyzers.

Analyzers read only the cached Parquet + manifest written by Stage 1; nothing
here touches EvoXBench.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def load_space(full_root: str, space: str):
    """Return (df, manifest) for a cached space."""
    sdir = os.path.join(full_root, space)
    df = pd.read_parquet(os.path.join(sdir, 'samples.parquet'))
    with open(os.path.join(sdir, 'manifest.json')) as fh:
        manifest = json.load(fh)
    return df, manifest


def metric_cols(manifest) -> list[str]:
    return list(manifest['metrics'])


def decision_cols(df) -> list[str]:
    return [c for c in df.columns if c.startswith('x') and c[1:].isdigit()]


def out_dir(full_root: str, space: str) -> str:
    d = os.path.join(full_root, space)
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path)
    else:
        pd.DataFrame(obj).to_csv(path, index=False)


def write_tex(df: pd.DataFrame, path: str, caption: str, label: str,
              float_format: str = '%.4g', index: bool = True) -> None:
    """Write a booktabs LaTeX table; numeric-only escaping kept minimal."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = df.to_latex(
        float_format=lambda v: (float_format % v) if np.isfinite(v) else 'nan',
        index=index, escape=True, na_rep='--', longtable=False,
    )
    tex = (
        '\\begin{table}[htbp]\n\\centering\n\\small\n'
        f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
        f'{body}'
        '\\end{table}\n'
    )
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(tex)


def safe_log10(x: np.ndarray):
    """log10 with a strict-positivity report. Returns (values_or_None, n_nonpos)."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n_nonpos = int(np.sum(finite & (x <= 0)))
    if n_nonpos > 0:
        return None, n_nonpos
    return np.log10(x), 0
