# IOC-SAMO-COBRA — vendored source

Upstream: <https://github.com/RoydeZomer/IOC-SAMO-COBRA> @ `d2155e4a` (2025-05-15), MIT (see `LICENCE`).

## Why vendored rather than declared a dependency

It cannot be installed. Upstream is a flat script repo — no `setup.py`, no
`pyproject.toml`, no package directory, no `__init__.py` — whose modules import
each other by bare name (`from SACOBRA import plog`), assuming the interpreter
runs from inside that directory (`start_pSAMO-COBRA_cheap.py`). It is not on
PyPI under any name. Even a hand-written `setup.py` would be the wrong answer:
it would install ten top-level modules called `lhs`, `halton`, `hypervolume`,
`SACOBRA`, `RbfInter`, `transformLHS` and `paretofrontFeasible` into
site-packages, where several are obvious collision hazards.

This differs from `pysamoo` / `robustranking` / `SMAC3`, which are sibling
checkouts installed with `pip install -e` because they *are* real packages.

## What was copied

Nine of upstream's ten modules — the transitive closure of the two entry points
`cheap_SAMO_COBRA_Init` and `cheap_SAMO_COBRA_PhaseII`:

| file | role |
| --- | --- |
| `cheap_SAMO_COBRA_Init.py` | entry point 1: initial design, `cobra` state dict |
| `cheap_SAMO_COBRA_PhaseII.py` | entry point 2: the optimisation loop |
| `SACOBRA.py` | scaling / rescaling / plog helpers |
| `RbfInter.py` | RBF surrogate fit and interpolation |
| `lhs.py`, `halton.py`, `transformLHS.py` | initial-design generators |
| `paretofrontFeasible.py` | feasible non-dominated filter |
| `hypervolume.py` | HV indicator used by the PHV infill criterion |

`visualiseParetoFront.py` was **not** copied: a matplotlib/mplot3d live-plot
helper with exactly one call site, guarded by `cobra['plot']`, which a batch
campaign never sets.

## Modifications

Every change is marked in-place with a `VENDORED,` comment.

**Carried over from the `SAMOS-NAS` copy** (already divergent from upstream, and
both matter here):

1. `SACOBRA.rescale_constr` — guard `divider == 0` (identity scaling). Upstream
   divides by the constraint's observed range; a constraint that is constant
   across the initial design gives 0. Reaching a constant constraint column is
   plausible on a rounded integer NAS space.
2. `cheap_SAMO_COBRA_PhaseII` — return a large finite value when the
   non-dominated sort yields an empty first front, and `np.max(..., initial=0)`
   so an empty penalty set does not raise.

**New here:**

3. **pygmo → pymoo.** Upstream's only two pygmo calls were
   `pygmo.hypervolume` (`hypervolume.py`) and `pygmo.fast_non_dominated_sorting`
   (`cheap_SAMO_COBRA_PhaseII.py`). pygmo is a conda-only build of the pagmo2
   C++ library and is not otherwise a dependency of this repo; pymoo is, and
   computes both quantities identically. This removes the dependency outright
   rather than gambling on pygmo being present in the cluster env.
4. **`visualiseParetoFront` import removed** (see above).
5. **End-of-run disk dump made opt-in** — `saveResults=False` on
   `cheap_SAMO_COBRA_Init`, guarding the block in `cheap_SAMO_COBRA_PhaseII`
   that writes six CSVs and a JSON to `./batchresults/<name>/batch<N>/`
   relative to the *current working directory*. Two problems, both fixed by the
   switch: it litters whatever directory the run starts in, and `<name>` comes
   from `str(fn).split(' ')[6]` — parsing the objective function's `repr`, which
   `IndexError`s on any repr with fewer than seven space-separated tokens (a
   `functools.partial`, a lambda, many bound methods). That is a latent crash on
   the way *out* of an otherwise completed run.
6. **Bare imports rewritten as relative imports**, making this an ordinary
   subpackage. Upstream's `SAMOS-NAS` copy instead shipped an `__init__.py` that
   inserted its own directory into `sys.path`; that works, but publishes
   `lhs`, `hypervolume`, `SACOBRA` etc. as importable top-level names.
7. `SACOBRA.standardize_obj` — guard `std == 0` (divide by 1.0 instead), the
   objective-side twin of modification 1. A constant objective column is
   reachable on a rounded integer NAS space: a tie plateau, an all-invalid
   design whose metrics are guarded to a constant, or — under the scenario_run
   hard gate — an initial design with no feasible point, whose objective rows
   are all imputed with the same reference point. Upstream divides regardless
   and every surrogate downstream is then fitted on nan.

## Who wires this up

`experiments2/scenario_run/_ioccobra.py` — the scenario_run campaign's
`ioc-cobra` method. It supplies the duck-typed problem (backed by the campaign's
own `ConstrainedEvoXBenchProblem`, so rounding, normalisation, the `G` formula
and the hard gate are shared with every other method) and a recording shim that
drives the campaign callback, since COBRA never reaches pymoo's `minimize`. Read
its module docstring before changing anything here.

## Notes for whoever wires this up

- `scipy.optimize._cobyla` was removed in SciPy ≥ 1.10 and this env has 1.17.1.
  The import is already shimmed to `None`; its only use site is commented out
  upstream, and a full run verifies clean, so the shim is genuinely dead code.
- Pass `nCores=1`. Above 1, `cheap_SAMO_COBRA_Init` opens an
  `multiprocessing.Pool`.
- COBRA searches a **continuous** box. Any NAS adapter must round `x` before
  calling `benchmark.evaluate`, and should expect the RBF surrogate to see
  duplicate lattice points on small tabular spaces.
- The budget knob is `feval` (total real evaluations). COBRA drives its own
  loop, so it does not go through pymoo's `minimize`/`Evaluator` and needs its
  own results-recording path.
