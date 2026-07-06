"""
LaTeX Table Generator for NASBench201 Objective Sweep Results

Generates publication-ready LaTeX tables combining HV and IGD+ results
across multiple datasets with abbreviated objective names and tolerances.
"""

from pathlib import Path
import numpy as np

try:
    from scipy.stats import ranksums
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def generate_latex_table_all_datasets(
    datasets: list,
    hv_data_dict: dict,
    igd_data_dict: dict,
    runs_df,
    compute_final_statistics_fn,
    true_hv_fn,
    first_objective: str,
    second_objectives: list,
    all_data=None,
    save_dir: Path = None,
    output_filename: str = "combined_all_datasets_table.tex",
):
    """
    Generate a single LaTeX table with all datasets stacked vertically.
    HV and IGD+ are shown in separate sections for each dataset.
    Table is formatted for rotation (landscape) due to many columns.

    Parameters
    ----------
    datasets : list
        List of dataset names
    hv_data_dict : dict
        HV trajectory data
    igd_data_dict : dict
        IGD+ trajectory data
    runs_df : pd.DataFrame
        Runs metadata
    compute_final_statistics_fn : callable
        Function to compute final statistics
    true_hv_fn : callable
        Function to compute true Pareto HV
    first_objective : str
        Name of first objective (e.g., "epoch-200")
    second_objectives : list
        List of second objective names
    all_data : dict, optional
        Full benchmark data for computing global best HV
    save_dir : Path, optional
        Directory to save the .tex file
    output_filename : str
        Name of output .tex file

    Returns
    -------
    str
        Generated LaTeX code
    """
    # Collect statistics for all datasets
    all_dataset_stats = []

    for dataset in datasets:
        hv_stats = compute_final_statistics_fn("hv", dataset, hv_data_dict, runs_df)
        igd_stats = compute_final_statistics_fn("igd", dataset, igd_data_dict, runs_df)

        if not hv_stats and not igd_stats:
            print(f"  [!] No statistics for {dataset} - skipping")
            continue

        # Compute global best HV
        global_best_hv = {}
        if all_data is not None and hv_stats:
            objectives = sorted(hv_stats.keys())
            for obj in objectives:
                try:
                    hv_val = true_hv_fn(
                        dataset=dataset,
                        objectives=(first_objective, obj),
                        all_data=all_data,
                        normalized=True,
                    )
                    global_best_hv[obj] = hv_val
                except Exception:
                    pass

        all_dataset_stats.append({
            'dataset': dataset,
            'hv_stats': hv_stats,
            'igd_stats': igd_stats,
            'global_best_hv': global_best_hv,
        })

    if not all_dataset_stats:
        print(f"  [!] No statistics for any dataset - skipping LaTeX table")
        return

    # Get all methods and objectives (from first dataset with data)
    first_stats = all_dataset_stats[0]
    all_methods = set()
    for stats in [first_stats['hv_stats'], first_stats['igd_stats']]:
        if stats:
            for obj_stats in stats.values():
                all_methods.update(obj_stats.keys())

    # Method ordering: Random, Baseline, then surrogates
    method_order = []
    if "Random" in all_methods:
        method_order.append("Random")
    if "Baseline" in all_methods:
        method_order.append("Baseline")
    surrogate_methods = sorted([m for m in all_methods if m not in ["Random", "Baseline"]])
    method_order.extend(surrogate_methods)

    # Get objectives (use HV objectives as primary)
    objectives = sorted(first_stats['hv_stats'].keys()) if first_stats['hv_stats'] else sorted(first_stats['igd_stats'].keys())

    # Build LaTeX table
    lines = []
    lines.append("\\begin{table}[p]")  # Use 'p' for full page
    lines.append("\\centering")

    # Caption and label at the top
    dataset_names = ", ".join([ds.replace("_", "\\_") for ds in datasets])
    lines.append(f"\\caption{{Performance comparison across all datasets ({dataset_names}). "
                f"HV (higher is better) and IGD+ (lower is better) values shown. "
                f"Best method per metric bolded. Global Best shows true Pareto HV.}}")
    lines.append(f"\\label{{tab:combined_all_datasets}}")

    lines.append("\\begin{sideways}")
    lines.append("\\tiny")  # Smaller font for multiple datasets

    # Table structure - one column per objective
    col_spec = "l" + "r" * len(objectives)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Abbreviate objective names to save space
    obj_abbrev = {
        'edgegpu_latency': 'edge-lat',
        'edgegpu_energy': 'edge-ene',
        'raspi4_latency': 'raspi-lat',
        'pixel3_latency': 'pix3-lat',
        'eyeriss_latency': 'eyrs-lat',
        'eyeriss_energy': 'eyrs-ene',
        'eyeriss_arithmetic_intensity': 'eyrs-ai',
        'fpga_latency': 'fpga-lat',
        'fpga_energy': 'fpga-ene',
        'flops': 'flops',
        'params': 'params',
    }

    header1 = "Method"
    for obj in objectives:
        # Use abbreviated names
        obj_name = obj_abbrev.get(obj, obj.replace("_", "-")[:8])
        header1 += f" & {obj_name}"
    header1 += " \\\\"
    lines.append(header1)
    lines.append("\\midrule")

    # Process each dataset - HV section then IGD+ section
    for idx, ds_stats in enumerate(all_dataset_stats):
        dataset = ds_stats['dataset']
        hv_stats = ds_stats['hv_stats']
        igd_stats = ds_stats['igd_stats']
        global_best_hv = ds_stats['global_best_hv']

        # Find best methods for this dataset (all methods within rounding of best)
        hv_best = {}   # obj -> set of method names
        igd_best = {}  # obj -> set of method names

        if hv_stats:
            for obj in objectives:
                if obj in hv_stats:
                    valid = {m: hv_stats[obj][m][0] for m in hv_stats[obj]}
                    if valid:
                        best_val = max(valid.values())
                        best_rounded = round(best_val, 3)
                        hv_best[obj] = {m for m, v in valid.items() if round(v, 3) >= best_rounded}

        if igd_stats:
            for obj in objectives:
                if obj in igd_stats:
                    valid = {m: igd_stats[obj][m][0] for m in igd_stats[obj]}
                    if valid:
                        best_val = min(valid.values())
                        best_rounded = round(best_val, 3)
                        igd_best[obj] = {m for m, v in valid.items() if round(v, 3) <= best_rounded}

        # Dataset header row - HV section
        dataset_clean = dataset.replace("_", "\\_")
        n_cols = 1 + len(objectives)
        lines.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{dataset_clean} -- Hypervolume (HV)}}}} \\\\")
        lines.append("\\midrule")

        # HV data rows for this dataset
        for method in method_order:
            method_clean = method.replace("_", "\\_")
            row = method_clean

            for obj in objectives:
                if hv_stats and obj in hv_stats and method in hv_stats[obj]:
                    mean, std = hv_stats[obj][method]
                    value_str = f"{mean:.3f}$_{{{{{std:.3f}}}}}$"
                    if obj in hv_best and method in hv_best[obj]:
                        value_str = f"\\textbf{{{value_str}}}"
                    row += f" & {value_str}"
                else:
                    row += " & --"

            row += " \\\\"
            lines.append(row)

        # Global best row for HV
        if global_best_hv:
            lines.append("\\midrule")
            row = "Global Best"
            for obj in objectives:
                if obj in global_best_hv:
                    row += f" & {global_best_hv[obj]:.3f}"
                else:
                    row += " & --"
            row += " \\\\"
            lines.append(row)

        # Separator and IGD+ section header
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{dataset_clean} -- IGD+}}}} \\\\")
        lines.append("\\midrule")

        # IGD+ data rows for this dataset
        for method in method_order:
            method_clean = method.replace("_", "\\_")
            row = method_clean

            for obj in objectives:
                if igd_stats and obj in igd_stats and method in igd_stats[obj]:
                    mean, std = igd_stats[obj][method]
                    if mean < 0.01:
                        value_str = f"{mean:.2e}$_{{{{{std:.2e}}}}}$"
                    else:
                        value_str = f"{mean:.3f}$_{{{{{std:.3f}}}}}$"
                    if obj in igd_best and method in igd_best[obj]:
                        value_str = f"\\textbf{{{value_str}}}"
                    row += f" & {value_str}"
                else:
                    row += " & --"

            row += " \\\\"
            lines.append(row)

        # Separator between datasets (not after last one)
        if idx < len(all_dataset_stats) - 1:
            lines.append("\\midrule")
            lines.append("\\midrule")  # Double line for visual separation

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{sideways}")
    lines.append("\\end{table}")

    # Save to file
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[OK] Saved combined LaTeX table: {out_path}")

    return "\n".join(lines)


# ─── NASBench-101 baseline table ──────────────────────────────────────────────

def generate_latex_table_nasbench101(
    methods: list,
    n_gen: int,
    results_root: str,
    labels: dict = None,
    save_dir: Path = None,
    output_filename: str = "nasbench101_baseline_table.tex",
) -> str:
    """Generate a publication-ready LaTeX table of final HV and IGD+ statistics
    for NASBench-101 baseline methods.

    Each row is one method; columns are: Method | HV mean±std | IGD+ mean±std.
    The best HV (highest) and best IGD+ (lowest) values are bold.

    Parameters
    ----------
    methods:         ordered list of method keys (must match result sub-directories)
    n_gen:           number of outer generations used in the runs (used to
                     identify the final-generation entry in each seed file)
    results_root:    root directory containing per-method result folders
    labels:          optional dict mapping method keys to display names
    save_dir:        if provided, write the .tex file here
    output_filename: name of the output .tex file

    Returns
    -------
    str   LaTeX source of the table
    """
    import os
    import pickle

    _labels = {
        'random':       'Random',
        'nsga2':        'NSGA-II (2-pt XO, unif.\\ mut.)',
        'nsga2-single': 'NSGA-II (no XO, single-pt mut.)',
        'samos-rfr':    'SAMOS (RFR surrogate)',
        'samos-xgb':    'SAMOS (XGBoost surrogate)',
        'mosmac': "MO-SMAC"
    }
    if labels:
        _labels.update(labels)

    # ── collect final-generation statistics per method ─────────────────────

    stats = {}   # method -> {'hv': (mean, std), 'igd': (mean, std), 'n_seeds': int}

    for method in methods:
        seed_dir = os.path.join(results_root, method)
        if not os.path.isdir(seed_dir):
            stats[method] = None
            continue

        hv_finals, igd_finals = [], []
        for pkl_file in sorted(os.listdir(seed_dir)):
            if not pkl_file.endswith('.pkl'):
                continue
            with open(os.path.join(seed_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)

            indicators = data.get('indicators', [])
            if not indicators:
                continue

            # Final-generation entry: list may be one-per-gen or one-per-individual
            if len(indicators) >= n_gen:
                step = len(indicators) // n_gen
                final = indicators[min(n_gen * step - 1, len(indicators) - 1)]
            else:
                final = indicators[-1]

            hv_finals.append(final.get('hv', 0.0))
            igd_finals.append(final.get('igd_plus', np.nan))

        if hv_finals:
            stats[method] = {
                'hv':  (float(np.mean(hv_finals)),  float(np.std(hv_finals))),
                'igd': (float(np.nanmean(igd_finals)), float(np.nanstd(igd_finals))),
                'n_seeds': len(hv_finals),
            }
        else:
            stats[method] = None

    # ── identify best values for bold formatting ───────────────────────────

    hv_means  = [stats[m]['hv'][0]  for m in methods if stats.get(m)]
    igd_means = [stats[m]['igd'][0] for m in methods if stats.get(m)]
    best_hv   = max(hv_means)  if hv_means  else None
    best_igd  = min(igd_means) if igd_means else None

    # ── build LaTeX ────────────────────────────────────────────────────────

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{NASBench-101 baseline results at final generation "
        "(test\\textsubscript{acc@108} $\\times$ $n_{\\text{params}}$). "
        "Mean $\\pm$ std over seeds. "
        "\\textbf{Bold} = best per column.}",
        "  \\label{tab:nasbench101_baseline}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Method & Seeds & HV $\\uparrow$ & IGD+ $\\downarrow$ \\\\",
        "    \\midrule",
    ]

    for method in methods:
        s = stats.get(method)
        label = _labels.get(method, method).replace('_', '\\_')

        if s is None:
            lines.append(f"    {label} & -- & -- & -- \\\\")
            continue

        hv_str  = f"{s['hv'][0]:.4f} $\\pm$ {s['hv'][1]:.4f}"
        igd_str = f"{s['igd'][0]:.4f} $\\pm$ {s['igd'][1]:.4f}"

        if best_hv  is not None and abs(s['hv'][0]  - best_hv)  < 1e-9:
            hv_str  = f"\\textbf{{{hv_str}}}"
        if best_igd is not None and abs(s['igd'][0] - best_igd) < 1e-9:
            igd_str = f"\\textbf{{{igd_str}}}"

        lines.append(f"    {label} & {s['n_seeds']} & {hv_str} & {igd_str} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"[OK] Saved NASBench-101 LaTeX table: {out_path}")

    return tex


# ─── obj-GA sweep table ───────────────────────────────────────────────────────

def _format_mean_std(mean: float, std: float, sci: bool = True) -> str:
    """Format a ``mean$_{std}$`` cell, mirroring the existing table style.

    With *sci* (default), switches to scientific notation for very small
    magnitudes (as the IGD+ branch of
    :func:`generate_latex_table_all_datasets` already does).  Pass ``sci=False``
    to always use fixed 3-decimal notation -- keeps cells narrow when tiny values
    would otherwise expand to ``e-04`` form.
    """
    if not np.isfinite(mean):
        return "--"
    if sci and abs(mean) < 0.01 and mean != 0.0:
        return f"{mean:.2e}$_{{{{{std:.2e}}}}}$"
    return f"{mean:.3f}$_{{{{{std:.3f}}}}}$"


def _holm_correct(p_values: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni correction. Returns bool list (True = reject H0).

    Identical logic to ``pairwise_significance_test._holm_correct``.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    reject = [False] * n
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (n - rank):
            reject[idx] = True
        else:
            break
    return reject


def _checkpoint_samples(arr, gen_idx: int):
    """Return per-seed values at *gen_idx*, or None when unavailable.

    Accepts either a 1-D mean trajectory ``(n_gen,)`` (no per-seed info → None)
    or a 2-D per-seed matrix ``(n_seeds, n_gen)``.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        col = min(gen_idx, a.shape[1] - 1)
        return a[:, col]
    return None


def _checkpoint_mean_std(arr, gen_idx: int):
    """Return ``(mean, std)`` at *gen_idx* for a 1-D mean or 2-D per-seed array."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        col = min(gen_idx, a.shape[1] - 1)
        vals = a[:, col]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan, 0.0
        return float(np.mean(vals)), float(np.std(vals))
    idx = min(gen_idx, len(a) - 1)
    return float(a[idx]), 0.0


def generate_obj_ga_table(
    data_dict: dict,
    methods: list,
    method_labels: dict,
    problems: list,
    problem_labels: dict,
    metric: str,
    checkpoint_gens: list,
    out_path,
    baseline_method: str = "nsga2",
    alpha: float = 0.05,
) -> str:
    """Generate a LaTeX table of HV or IGD+ at several generation checkpoints.

    One table for a single (benchmark, n_obj_group); call once per metric and
    write to a separate file each time. Layout:
    - Columns: Methods (algorithms)
    - Rows: Problems (grouped), with sub-rows for each checkpoint generation

    Cells show ``mean$_{std}$``; the best value per (problem, generation) is bold;
    a significance marker (``^{*}``) is appended where the method differs
    significantly from *baseline_method* under a Holm-corrected Wilcoxon rank-sum test.

    Parameters
    ----------
    data_dict
        ``{problem: {method: {'hv': arr, 'igd_plus': arr}}}`` where each ``arr``
        is either a 1-D mean trajectory ``(n_gen,)`` or a 2-D per-seed matrix
        ``(n_seeds, n_gen)``.  2-D arrays enable Wilcoxon significance markers.
    methods
        Ordered method keys (table column order).
    method_labels
        ``{method: display label}``.
    problems
        Ordered problem keys (must be keys of *data_dict*).
    problem_labels
        ``{problem: display label}``.
    metric
        ``'hv'`` (higher is better) or ``'igd_plus'`` (lower is better).
    checkpoint_gens
        1-based generation numbers to show as rows (``config.metric_checkpoints``).
    out_path
        Path to the ``.tex`` file to write.
    baseline_method
        Method against which Wilcoxon tests are run (default ``'nsga2'``).
    alpha
        Significance level for the Holm-corrected tests (default 0.05).

    Returns
    -------
    str   The LaTeX source.
    """
    metric_key = 'hv' if metric.lower() in ('hv', 'hypervolume') else 'igd_plus'
    higher_is_better = (metric_key == 'hv')
    metric_disp = 'HV $\\uparrow$' if higher_is_better else 'IGD+ $\\downarrow$'

    # checkpoint_gens are 1-based generations → 0-based array indices
    ckpt_idx = [g - 1 for g in checkpoint_gens]

    # ── column spec: Gen / Problem | method1 | method2 | ... ───────────────
    n_methods = len(methods)
    col_spec = "ll" + "r" * n_methods

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{metric_disp.split(' ')[0]} at generation checkpoints "
        f"({', '.join(str(g) for g in checkpoint_gens)}). "
        f"Mean$_{{\\text{{std}}}}$ over seeds. \\textbf{{Bold}} = best per (generation, problem). "
        f"$^{{*}}$ = significant vs.\\ {method_labels.get(baseline_method, baseline_method)} "
        f"(Holm-corrected Wilcoxon, $\\alpha={alpha}$).}}",
        "  \\label{tab:obj_ga_" + metric_key + "}",
        f"  \\resizebox{{\\textwidth}}{{!}}{{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
    ]

    # ── header row: method labels ────────────────────────────────────────────
    h = "    Gen & Problem"
    for m in methods:
        mlabel = method_labels.get(m, m).replace('_', '\\_')
        h += f" & {mlabel}"
    h += " \\\\"
    lines.append(h)
    lines.append("    \\midrule")

    # ── determine best value per (problem, checkpoint) ──────────────────────
    # best_vals[(prob, gen_idx)] = best mean value among methods
    best_vals = {}
    for prob in problems:
        pdata = data_dict.get(prob, {})
        for gi in ckpt_idx:
            col_means = []
            for m in methods:
                if m in pdata and metric_key in pdata[m]:
                    mean, _ = _checkpoint_mean_std(pdata[m][metric_key], gi)
                    if np.isfinite(mean):
                        col_means.append(mean)
            if col_means:
                best_vals[(prob, gi)] = (max(col_means) if higher_is_better
                                         else min(col_means))

    # ── precompute Holm-corrected significance per (problem, checkpoint) ────
    sig = {}
    if _SCIPY_AVAILABLE:
        compare_methods = [m for m in methods if m != baseline_method]
        for prob in problems:
            pdata = data_dict.get(prob, {})
            base = pdata.get(baseline_method, {})
            base_arr = base.get(metric_key)
            for gi in ckpt_idx:
                base_samp = _checkpoint_samples(base_arr, gi) if base_arr is not None else None
                if base_samp is None:
                    continue
                base_samp = base_samp[np.isfinite(base_samp)]
                if len(base_samp) < 3:
                    continue
                pvals, order = [], []
                for m in compare_methods:
                    arr = pdata.get(m, {}).get(metric_key)
                    samp = _checkpoint_samples(arr, gi) if arr is not None else None
                    if samp is None:
                        continue
                    samp = samp[np.isfinite(samp)]
                    if len(samp) < 3:
                        continue
                    try:
                        _, p = ranksums(base_samp, samp)
                    except Exception:
                        p = 1.0
                    pvals.append(float(p))
                    order.append(m)
                if not pvals:
                    continue
                reject = _holm_correct(pvals, alpha)
                sig[(prob, gi)] = {m: r for m, r in zip(order, reject)}

    # ── data rows: grouped by generation, with sub-rows per problem ───────
    first_gen = True
    for gi_idx, gi in enumerate(ckpt_idx):
        g = checkpoint_gens[gi_idx]

        for prob_idx, prob in enumerate(problems):
            pdata = data_dict.get(prob, {})
            plabel = problem_labels.get(prob, prob).replace('_', '\\_')

            # First problem of each generation: show generation label
            if prob_idx == 0:
                if not first_gen:
                    lines.append("    \\addlinespace")
                first_gen = False
                row = f"    Gen {g} & {plabel}"
            else:
                row = f"     & {plabel}"

            for m in methods:
                mdata = pdata.get(m, {})
                arr = mdata.get(metric_key)
                if arr is None:
                    row += " & --"
                    continue
                mean, std = _checkpoint_mean_std(arr, gi)
                cell = _format_mean_std(mean, std)
                best = best_vals.get((prob, gi))
                if best is not None and np.isfinite(mean) and abs(mean - best) < 1e-9:
                    cell = f"\\textbf{{{cell}}}"
                if sig.get((prob, gi), {}).get(m, False):
                    cell = f"{cell}$^{{*}}$"
                row += f" & {cell}"

            row += " \\\\"
            lines.append(row)

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "\\end{table}",
    ]

    tex = "\n".join(lines)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] Saved obj-GA {metric_key} LaTeX table: {out_path}")

    return tex


def generate_obj_ga_nobj_comparison_table(
    sections: list,
    methods: list,
    method_labels: dict,
    metric: str,
    final_gen: int,
    out_path,
    baseline_method: str = 'nsga2',
    alpha: float = 0.05,
) -> str:
    """Generate a single LaTeX table comparing algorithms at *final_gen* across
    all n_obj groups and benchmarks.

    Parameters
    ----------
    sections
        Ordered list of ``(n_obj, subsection_header, problem_keys, table_data,
        problem_labels, hv_ceilings)`` tuples, one per (n_obj, benchmark)
        combination.  ``hv_ceilings`` is ``{problem_key: float | None}``; HV
        values are normalised by this ceiling when available.
    methods
        Ordered method keys (table columns).
    method_labels
        ``{method: display label}``.
    metric
        ``'hv'`` or ``'igd_plus'``.
    final_gen
        1-based generation number shown (e.g. 100).
    out_path
        Destination ``.tex`` file path.
    baseline_method
        Method for Wilcoxon significance tests.
    alpha
        Significance level (Holm-corrected).

    Returns
    -------
    str   The LaTeX source.
    """
    metric_key = 'hv' if metric.lower() in ('hv', 'hypervolume') else 'igd_plus'
    higher_is_better = (metric_key == 'hv')
    normalize_hv = (metric_key == 'hv')
    metric_disp = (r'Normalised HV $\uparrow$' if normalize_hv
                   else r'IGD$^{+}$ $\downarrow$')
    gi = final_gen - 1   # 0-based array index

    # Group sections by n_obj, preserving encounter order (2 → 3 → 4 → …).
    nobj_groups: dict = {}
    nobj_order: list = []
    for sec in sections:
        n_obj = sec[0]
        rest = sec[1:]   # (subsection_header, keys, table_data, labels, hv_ceilings)
        if n_obj not in nobj_groups:
            nobj_groups[n_obj] = []
            nobj_order.append(n_obj)
        nobj_groups[n_obj].append(rest)

    n_methods = len(methods)
    n_cols = 1 + n_methods
    col_spec = 'l' + 'r' * n_methods

    header_cells = [r'\textbf{Problem}'] + [
        r'\textbf{' + method_labels.get(m, m).replace('_', r'\_') + r'}'
        for m in methods
    ]

    baseline_label = method_labels.get(baseline_method, baseline_method).replace('_', r'\_')
    norm_note = (r' HV is normalised by the Pareto-front HV (1\,=\,ideal).'
                 if normalize_hv else '')
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (
            r'\caption{' + metric_disp.split(r' $')[0]
            + r' at generation ' + str(final_gen)
            + r' (mean$_{\text{std}}$ over seeds).'
            + norm_note
            + r' \textbf{Bold}: best per row.'
            r' Superscripts vs.\ ' + baseline_label
            + r' (Holm-corrected Wilcoxon, $\alpha=' + str(alpha) + r'$):'
            r' $^{+}$\,better, $^{-}$\,worse, $^{\sim}$\,n.s.}'
        ),
        r'\label{tab:obj_ga_' + metric_key + r'_combined_nobj}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        ' & '.join(header_cells) + r' \\',
    ]

    for n_obj in nobj_order:
        obj_label = f'{n_obj} objectives'
        lines.append(r'\midrule')
        lines.append(
            r'\multicolumn{' + str(n_cols) + r'}{l}{\textbf{'
            + obj_label + r'}} \\'
        )

        for subsection_header, problem_keys, table_data, problem_labels, hv_ceilings in nobj_groups[n_obj]:
            if not problem_keys:
                continue
            lines.append(r'\midrule')
            lines.append(
                r'\multicolumn{' + str(n_cols) + r'}{l}{\small\textit{'
                + subsection_header + r'}} \\'
            )
            lines.append(r'\midrule')

            for prob in problem_keys:
                pdata = table_data.get(prob, {})
                plabel = problem_labels.get(prob, prob).replace('_', r'\_')
                ceiling = (hv_ceilings.get(prob) if hv_ceilings else None) if normalize_hv else None
                valid_ceiling = ceiling and np.isfinite(ceiling) and ceiling > 0

                # Best value in this row (normalised where possible).
                row_means = {}
                for m in methods:
                    arr = pdata.get(m, {}).get(metric_key)
                    if arr is not None:
                        mean, _ = _checkpoint_mean_std(arr, gi)
                        if np.isfinite(mean):
                            row_means[m] = mean / ceiling if valid_ceiling else mean
                best_val = (
                    max(row_means.values()) if higher_is_better else min(row_means.values())
                ) if row_means else None

                # Directional Holm-corrected Wilcoxon vs baseline:
                # '+' = significantly better, '-' = worse, '~' = not significant.
                row_markers: dict = {}
                if _SCIPY_AVAILABLE:
                    base_arr = pdata.get(baseline_method, {}).get(metric_key)
                    base_samp = _checkpoint_samples(base_arr, gi) if base_arr is not None else None
                    if base_samp is not None:
                        base_samp = base_samp[np.isfinite(base_samp)]
                        if len(base_samp) >= 3:
                            compare_ms = [m for m in methods if m != baseline_method]
                            pvals, tested, samp_list = [], [], []
                            for m in compare_ms:
                                arr = pdata.get(m, {}).get(metric_key)
                                samp = _checkpoint_samples(arr, gi) if arr is not None else None
                                if samp is None:
                                    continue
                                samp = samp[np.isfinite(samp)]
                                if len(samp) < 3:
                                    continue
                                try:
                                    _, p = ranksums(base_samp, samp)
                                    pvals.append(float(p))
                                    tested.append(m)
                                    samp_list.append(samp)
                                except Exception:
                                    pass
                            if pvals:
                                reject = _holm_correct(pvals, alpha)
                                base_mean = float(np.mean(base_samp))
                                for m, r, samp in zip(tested, reject, samp_list):
                                    other_mean = float(np.mean(samp))
                                    if r:
                                        better = (other_mean > base_mean if higher_is_better
                                                  else other_mean < base_mean)
                                        row_markers[m] = '+' if better else '-'
                                    else:
                                        row_markers[m] = '~'

                cells = [plabel]
                for m in methods:
                    arr = pdata.get(m, {}).get(metric_key)
                    if arr is None:
                        cells.append('--')
                        continue
                    mean, std = _checkpoint_mean_std(arr, gi)
                    if valid_ceiling:
                        mean = mean / ceiling
                        std  = std  / ceiling
                    cell = _format_mean_std(mean, std)
                    if best_val is not None and np.isfinite(mean) and abs(mean - best_val) < 1e-9:
                        cell = r'\textbf{' + cell + r'}'
                    marker = row_markers.get(m)
                    if marker == '+':
                        cell += r'$^{+}$'
                    elif marker == '-':
                        cell += r'$^{-}$'
                    elif marker == '~':
                        cell += r'$^{\sim}$'
                    cells.append(cell)

                lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}',
    ]

    tex = '\n'.join(lines)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f'[OK] Saved combined n_obj {metric_key} table: {out_path}')

    return tex


# ─── combined Experiment 1 (full, no predictor) vs Experiment 2 (GA+predictor) ──

_PRED_ABBREV = {'xgboost': 'XGB', 'rf': 'RF', 'rbf': 'RBF'}


def _final_hv_mean_std(arr, ceiling=None):
    """Return ``(mean, std)`` of a 1-D per-seed final-gen HV array.

    When *ceiling* is a finite positive number the values are normalised by it
    (so ``1`` is the pooled-Pareto-front HV).  Returns ``(nan, 0.0)`` for empty
    or all-non-finite input.
    """
    if arr is None:
        return np.nan, 0.0
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    if ceiling is not None and np.isfinite(ceiling) and ceiling > 0:
        a = a / ceiling
    return float(np.mean(a)), float(np.std(a))


def generate_combined_exp1_exp2_table(
    sections: list,
    gas: list,
    ga_labels: dict,
    predictors: list,
    predictor_labels: dict,
    n_obj_label: str,
    out_path,
    alpha: float = 0.05,
    gas_per_table: int = 4,
    variant_note: str = '',
    label_suffix: str = '',
    baseline_ga: str = 'random',
    sig_baseline_ga: str = None,
) -> str:
    """Generate one table per objective-count group comparing every GA x
    predictor *surrogate* combination (Experiment 2) against a single reference:
    full-budget random search with no predictor (the Experiment-1 ``random``
    run, shown once as the ``Rand (full)`` column).

    The GAs are stacked vertically in panels of *gas_per_table* (default 4)
    inside one ``table*`` float; the ``Rand (full)`` reference column is repeated
    in each panel.  Each GA group shows one column per predictor (XGB/RF/RBF).
    Rows are problems, organised into benchmark subsections.  All cells show
    normalised HV (``mean$_{std}$``, higher is better; ``1`` = pooled-Pareto-front
    HV).  The best predictor *within* each GA per row is **bold**.  Every
    surrogate cell carries a directional, Holm-corrected Wilcoxon marker against
    ``Rand (full)`` (Holm family = all surrogate cells in the row):
    ``$^{+}$`` better, ``$^{-}$`` worse, ``$^{\\sim}$`` not significant.

    Parameters
    ----------
    sections
        Ordered list of ``(subsection_header, problem_keys, problem_labels,
        prob_data)`` tuples.  ``prob_data`` maps each problem key to
        ``{'ceiling': float | None, 'cols': {ga: {pred: arr, ..., 'full': arr}}}``
        where every ``arr`` is a 1-D per-seed final-generation HV array (raw,
        un-normalised) and ``'full'`` holds the Experiment-1 reference.
    gas
        Ordered GA keys (one column group each).
    ga_labels
        ``{ga: display label}``.
    predictors
        Ordered predictor keys (sub-columns within each GA group).
    predictor_labels
        ``{predictor: display label}`` (used only for the caption legend).
    n_obj_label
        Human label for the objective-count group, e.g. ``"2"`` or ``"4+"``.
    out_path
        Destination ``.tex`` file path.
    alpha
        Significance level (Holm-corrected).

    Returns
    -------
    str   The LaTeX source.
    """
    n_pred = len(predictors)
    group_cols = n_pred + 1                         # XGB/RF/RBF + per-GA Full
    pred_keys = list(predictors) + ['full']         # column keys within a GA group
    ga_groups = [g for g in gas if g != baseline_ga]  # baseline GA shown only as the reference
    ref_label = ga_labels.get(baseline_ga, baseline_ga).replace('_', r'\_')
    ref_col_head = ref_label + r' (full)'

    # Significance baseline (which Full column the markers test against) may differ
    # from the displayed reference column; defaults to the reference GA.
    if sig_baseline_ga is None:
        sig_baseline_ga = baseline_ga
    sig_label = ga_labels.get(sig_baseline_ga, sig_baseline_ga).replace('_', r'\_') + r' (full)'

    # ── caption legend mapping abbreviations -> full predictor names ────────
    legend = ', '.join(
        f"{_PRED_ABBREV.get(p, p.upper())}={predictor_labels.get(p, p)}"
        for p in predictors
    )

    def _baseline_samples(cols):
        """Per-seed final HV of the significance-baseline GA's full run."""
        arr = cols.get(sig_baseline_ga, {}).get('full')
        if arr is None:
            return None
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        return a if len(a) >= 3 else None

    def _group_best_val(gcols, ceiling):
        """Best normalised mean within a single GA's XGB/RF/RBF/Full columns."""
        vals = []
        for ckey in pred_keys:
            mean, _ = _final_hv_mean_std(gcols.get(ckey), ceiling)
            if np.isfinite(mean):
                vals.append(mean)
        return max(vals) if vals else None

    def _row_markers(cols):
        """Directional Holm-corrected Wilcoxon for every non-random cell in a row
        (each GA's XGB/RF/RBF/Full) vs. the single full-random baseline.  Returns
        ``{(ga, ckey): '+'/'-'/'~'}``; Holm family = all tested cells in the row."""
        out = {}
        if not _SCIPY_AVAILABLE:
            return out
        base = _baseline_samples(cols)
        if base is None:
            return out
        base_mean = float(np.mean(base))
        pvals, tested, means = [], [], []
        for ga in ga_groups:
            gcols = cols.get(ga, {})
            for ckey in pred_keys:
                if ga == sig_baseline_ga and ckey == 'full':
                    continue   # the baseline column itself -- no marker
                arr = gcols.get(ckey)
                if arr is None:
                    continue
                samp = np.asarray(arr, float)
                samp = samp[np.isfinite(samp)]
                if len(samp) < 3:
                    continue
                try:
                    _, pv = ranksums(base, samp)
                except Exception:
                    continue
                pvals.append(float(pv))
                tested.append((ga, ckey))
                means.append(float(np.mean(samp)))
        if not pvals:
            return out
        reject = _holm_correct(pvals, alpha)
        for key, r, m in zip(tested, reject, means):
            if r:
                out[key] = '+' if m > base_mean else '-'
            else:
                out[key] = '~'
        return out

    # Split the non-random GAs into panels of *gas_per_table*, stacked VERTICALLY
    # inside a single table (one float, one tabular).  A leading reference column
    # holds the full-random baseline and is repeated in every panel; the column
    # count is fixed by *gas_per_table* and short trailing panels are padded.
    chunks = [ga_groups[i:i + gas_per_table]
              for i in range(0, len(ga_groups), gas_per_table)]
    n_parts = len(chunks)
    base_label = n_obj_label.replace('+', 'plus')

    n_cols = 2 + gas_per_table * group_cols        # Problem + Rand(full) + groups
    col_spec = 'l|r' + ('|' + 'r' * group_cols) * gas_per_table

    def _pad(cells, chunk):
        """Pad a row's cells with blanks so every panel spans *gas_per_table* GAs."""
        missing = gas_per_table - len(chunk)
        return cells + [''] * (missing * group_cols)

    def _emit_panel(lines, chunk):
        """Append one GA-panel: its two header rows plus all problem rows."""
        missing = gas_per_table - len(chunk)

        # Top header: blank over Problem + reference col, one multicolumn per GA.
        top = ['', '']
        for ga in chunk:
            glab = ga_labels.get(ga, ga).replace('_', r'\_')
            top.append(r'\multicolumn{' + str(group_cols) + r'}{c|}{\textbf{'
                       + glab + r'}}')
        if missing:
            top.append(r'\multicolumn{' + str(missing * group_cols) + r'}{c}{}')
        lines.append(' & '.join(top) + r' \\')

        # Sub-header: Problem | <baseline> (full) | XGB/RF/RBF/Full per GA.
        sub = [r'\textbf{Problem}', r'\textit{' + ref_col_head + r'}']
        for _ga in chunk:
            for p in predictors:
                sub.append(_PRED_ABBREV.get(p, p.upper()))
            sub.append(r'\textit{Full}')
        lines.append(' & '.join(_pad(sub, chunk)) + r' \\')
        lines.append(r'\midrule')

        for subsection_header, problem_keys, problem_labels, prob_data in sections:
            if not problem_keys:
                continue
            lines.append(
                r'\multicolumn{' + str(n_cols) + r'}{l}{\textit{'
                + subsection_header.replace('_', r'\_') + r'}} \\'
            )
            lines.append(r'\midrule')

            for prob in problem_keys:
                entry = prob_data.get(prob, {})
                cols = entry.get('cols', {})
                ceiling = entry.get('ceiling')
                plabel = problem_labels.get(prob, prob).replace('_', r'\_')

                markers = _row_markers(cols)
                ref_mean, ref_std = _final_hv_mean_std(
                    cols.get(baseline_ga, {}).get('full'), ceiling)
                cells = [plabel, _format_mean_std(ref_mean, ref_std, sci=False)]

                for ga in chunk:
                    gcols = cols.get(ga, {})
                    best_val = _group_best_val(gcols, ceiling)
                    for ckey in pred_keys:
                        mean, std = _final_hv_mean_std(gcols.get(ckey), ceiling)
                        cell = _format_mean_std(mean, std, sci=False)
                        if (best_val is not None and np.isfinite(mean)
                                and abs(mean - best_val) < 1e-9):
                            cell = r'\textbf{' + cell + r'}'
                        mk = markers.get((ga, ckey))
                        if mk == '+':
                            cell += r'$^{+}$'
                        elif mk == '-':
                            cell += r'$^{-}$'
                        elif mk == '~':
                            cell += r'$^{\sim}$'
                        cells.append(cell)
                lines.append(' & '.join(_pad(cells, chunk)) + r' \\')

            lines.append(r'\midrule')

        if lines[-1] == r'\midrule':
            lines.pop()

    panels_note = (
        f' The {len(ga_groups)} non-random algorithms are split into {n_parts} '
        f'vertically stacked panels of up to {gas_per_table} algorithms each.'
        if n_parts > 1 else '')

    lines = [
        r'\begin{table*}[p]',
        r'\centering',
        (
            r'\caption{Normalised HV $\uparrow$ at generation 100. Each algorithm '
            r'shows its three predictor surrogates (XGB/RF/RBF) and its own full '
            r'100-generation no-predictor run (\textbf{Full}, Experiment 1); the '
            r'leading \textbf{' + ref_col_head + r'} column gives the Experiment-1 '
            r'full-budget no-predictor baseline for reference, on the '
            + n_obj_label + r'-objective problems.' + panels_note
            + r' Mean$_{\text{std}}$ over seeds; HV is normalised by the '
            r'pooled-Pareto-front HV (1\,=\,ideal). \textbf{Bold} marks the best '
            r"of an algorithm's four columns per row. Superscripts test each cell "
            r'vs.\ \textbf{' + sig_label + r'} (Holm-corrected Wilcoxon over all cells in '
            r'the row, $\alpha=' + str(alpha)
            + r'$): $^{+}$ better, $^{-}$ worse, $^{\sim}$ n.s. '
            r'Predictors: ' + legend + r'.'
            + ((' ' + variant_note) if variant_note else '') + r'}'
        ),
        r'\label{tab:combined\_exp1\_exp2\_hv\_' + base_label + label_suffix + r'}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
    ]

    for part_idx, chunk in enumerate(chunks):
        if part_idx > 0:
            # Separate stacked panels with a little space and a double rule.
            lines.append(r'\midrule')
            lines.append(r'\midrule')
        _emit_panel(lines, chunk)

    lines += [
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}',
    ]

    tex = '\n'.join(lines)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f'[OK] Saved combined exp1/exp2 HV table ({n_obj_label} obj, '
          f'{n_parts} stacked panel{"s" if n_parts != 1 else ""}): {out_path}')

    return tex
