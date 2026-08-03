"""Cache IO and table helpers shared by the Stage-2 analyzers.

Analyzers read only the cached Parquet + manifest written by Stage 1; nothing
here touches EvoXBench.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


# Analysis-time row filters for known evaluator degeneracies (raw parquet left
# intact). MoSegNAS: stage-1 depth = 0 (x0==0) makes EvoXBench's look-up return
# a constant sentinel for every hardware/complexity metric regardless of the
# other genes -- contradicts the benchmark's own decomposable-sum model, so
# those resource objectives are invalid. Drops ~1/3 of MoSegNAS samples.
ROW_FILTERS = {
    'MoSegNAS': lambda df: df['x0'] != 0,
}


def apply_row_filter(df: pd.DataFrame, space: str, verbose: bool = False) -> pd.DataFrame:
    """Drop known-invalid rows for `space`; contiguous RangeIndex afterwards."""
    f = ROW_FILTERS.get(space)
    if f is None:
        return df
    keep = f(df).to_numpy()
    if verbose:
        print(f'  [filter] {space}: dropping {(~keep).sum():,} of {len(df):,} '
              f'invalid rows ({100*(~keep).mean():.1f}%)')
    return df[keep].reset_index(drop=True)


def load_space(full_root: str, space: str):
    """Return (df, manifest) for a cached space, with invalid rows filtered."""
    sdir = os.path.join(full_root, space)
    df = pd.read_parquet(os.path.join(sdir, 'samples.parquet'))
    df = apply_row_filter(df, space, verbose=True)
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
    # pandas escapes cell/label text but NOT axis names; sanitize underscores
    # there so the auto-generated tables compile under pdflatex.
    df = df.copy()
    df = df.rename_axis(
        index=[None if n is None else str(n).replace('_', ' ') for n in df.index.names],
        columns=[None if n is None else str(n).replace('_', ' ') for n in df.columns.names],
    )
    body = df.to_latex(
        float_format=lambda v: (float_format % v) if np.isfinite(v) else 'nan',
        index=index, escape=True, na_rep='--', longtable=False,
    )
    caption = caption.replace('_', '\\_')
    tex = (
        '\\begin{table}[htbp]\n\\centering\n\\small\n'
        f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
        f'{body}'
        '\\end{table}\n'
    )
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(tex)


def save_categorical(path: str, labels: np.ndarray, categories: list[str], **extra) -> None:
    """Compactly persist a small-cardinality string array (int8 codes + the
    category list) plus arbitrary small extra arrays, for re-plotting a
    cached analysis without recomputing it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    code_of = {c: i for i, c in enumerate(categories)}
    codes = np.array([code_of[v] for v in labels], dtype=np.int8)
    np.savez_compressed(path, codes=codes, categories=np.asarray(categories, dtype=object), **extra)


def load_categorical(path: str):
    """Inverse of :func:`save_categorical`: returns ``(labels, extra_dict)``."""
    d = np.load(path, allow_pickle=True)
    labels = d['categories'][d['codes']]
    extra = {k: d[k] for k in d.files if k not in ('codes', 'categories')}
    return labels, extra


def safe_log10(x: np.ndarray):
    """log10 with a strict-positivity report. Returns (values_or_None, n_nonpos)."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n_nonpos = int(np.sum(finite & (x <= 0)))
    if n_nonpos > 0:
        return None, n_nonpos
    return np.log10(x), 0
