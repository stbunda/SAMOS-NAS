"""
NASBench-101 search space size analysis.

Answers four questions about the 26-variable integer encoding
(5 op genes ∈ {0,1,2}, 21 edge genes ∈ {0,1}):

  1. How many raw options exist given 26 variables?
  2. How many options are valid?
  3. How many 'valid' options lead to duplicate architectures?
  4. What is the probability of encountering duplicates?

Run directly:  python analyze_nasbench101_searchspace.py
"""

import sys
import io
from math import comb, exp

# Ensure Unicode output works on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── constants (mirror problem/nasbench101/utils.py) ──────────────────────────

N_OPS   = 5     # genes 0-4: operation labels ∈ {0, 1, 2}
N_EDGES = 21    # genes 5-25: upper-triangular edges of 7-node DAG ∈ {0, 1}
N_OP_VALUES  = 3   # conv3x3-bn-relu, conv1x1-bn-relu, maxpool3x3
MAX_EDGES    = 9   # NASBench-101 hard cap on active edges
UNIQUE_ARCHS = 423_624  # canonical architectures in the benchmark


# ── 1. raw search space ───────────────────────────────────────────────────────

ops_combinations  = N_OP_VALUES ** N_OPS          # 3^5 = 243
edge_combinations = 2 ** N_EDGES                  # 2^21 = 2,097,152
total_options     = ops_combinations * edge_combinations   # ~509.6 M


# ── 2. valid options ──────────────────────────────────────────────────────────

# Edge constraint: at most MAX_EDGES edges set to 1
valid_edge_vecs = sum(comb(N_EDGES, k) for k in range(MAX_EDGES + 1))  # 695,860

# Combined before connectivity check
before_connectivity = ops_combinations * valid_edge_vecs   # ~169.1 M

# After connectivity + dedup (graph isomorphism):
# Multiple raw vectors can map to the same canonical (pruned) architecture,
# so valid_before_dedup ≥ UNIQUE_ARCHS and ≤ before_connectivity.
# The exact pre-dedup valid count requires running the full enumeration
# (see build_nasbench101_lut in problem/nasbench101/utils.py).
valid_before_dedup_lower = UNIQUE_ARCHS          # each arch has ≥1 representative
valid_before_dedup_upper = before_connectivity   # upper bound (if all connected)


# ── 3. duplicates among valid options ────────────────────────────────────────

# A "duplicate" valid vector maps to a canonical architecture that another
# valid vector also maps to (graph isomorphism after node pruning).
dup_upper = valid_before_dedup_upper - UNIQUE_ARCHS
dup_frac_upper = dup_upper / valid_before_dedup_upper   # ~99.75 %


# ── 4. duplicate probability ──────────────────────────────────────────────────

def p_collision_birthday(n: int, n_unique: int = UNIQUE_ARCHS) -> float:
    """P(at least one duplicate pair) in a population of n valid individuals,
    assuming uniform sampling over the n_unique canonical architectures.

    NOTE: this is an approximation. Sampling is not uniform over canonical
    architectures — simple architectures (fewer active nodes) are overrepresented
    because more raw vectors map to them (see multiplicity analysis below).
    True collision probability is higher than this estimate.
    """
    return 1.0 - exp(-n * (n - 1) / (2 * n_unique))


p_unique_from_raw = UNIQUE_ARCHS / total_options   # P(unique arch | random raw sample)


# -- 5. multiplicity non-uniformity -------------------------------------------
#
# Multiplicity = number of raw vectors that map to the same canonical architecture.
# It is NOT uniform across architectures — it depends on the number of active
# intermediate nodes (k) that survive pruning.
#
# Source of multiplicity:
#   (a) Op multiplicity: the (5-k) op genes for pruned-away nodes are "don't-cares".
#       Any of 3 values at each dead node produces the same canonical form.
#       Op multiplicity = 3^(5-k)
#
#   (b) Edge multiplicity: edges incident to dead nodes can often be freely set
#       to 0 or 1 as long as those nodes remain off any valid path.
#       This is harder to compute analytically (depends on graph topology)
#       but also increases for simpler/sparser architectures.
#
# Combined lower bound on multiplicity by active node count k:
#
#   k=5 (all nodes active):  >=  3^0 =  1   (dense, fully-used graphs)
#   k=4:                     >=  3^1 =  3
#   k=3:                     >=  3^2 =  9
#   k=2:                     >=  3^3 = 27
#   k=1 (minimal arch):      >=  3^4 = 81   (simple bottleneck graphs)
#
# Consequence for random sampling:
#   Uniformly sampling raw valid vectors is BIASED toward simple canonical
#   architectures. A 1-node bottleneck is at least 81x more likely to be
#   sampled than an equivalent 5-node architecture (op contribution alone).
#   The birthday problem estimate above therefore UNDERESTIMATES true
#   collision probability for simple architectures and overestimates for complex ones.

def op_multiplicity(k: int) -> int:
    """Lower bound on raw-vector multiplicity for a canonical arch with k active
    intermediate nodes, counting only the don't-care op genes contribution."""
    return N_OP_VALUES ** (N_OPS - k)


# ── reporting ─────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "─" * 60

    print(sep)
    print("NASBench-101 Search Space Analysis")
    print(sep)

    print("\n── 1. Raw search space ──────────────────────────────────")
    print(f"  Op genes   : {N_OPS} × {N_OP_VALUES} choices  → {ops_combinations:>12,}")
    print(f"  Edge genes : {N_EDGES} × 2 choices     → {edge_combinations:>12,}")
    print(f"  Total      : {ops_combinations} × {edge_combinations:,}  = {total_options:>12,}")

    print("\n── 2. Valid options ─────────────────────────────────────")
    print(f"  Edge vecs with ≤{MAX_EDGES} edges     : {valid_edge_vecs:>12,}")
    print(f"  Before connectivity filter : {before_connectivity:>12,}")
    print(f"  Unique valid architectures : {UNIQUE_ARCHS:>12,}")
    print(f"  (exact pre-dedup valid count requires running build_nasbench101_lut)")

    print("\n── 3. Duplicates among valid options ────────────────────")
    print(f"  Upper bound valid vecs  : {valid_before_dedup_upper:>12,}")
    print(f"  Unique architectures    : {UNIQUE_ARCHS:>12,}")
    print(f"  Max duplicate vecs      : {dup_upper:>12,}")
    print(f"  Duplicate fraction      : {dup_frac_upper*100:>11.4f}%  (upper bound)")

    print("\n-- 4. Duplicate probability")
    print(f"  P(unique arch | raw sample)  = {p_unique_from_raw*100:.5f}%")
    print()
    print("  Birthday problem -- P(at least one collision) in population of N:")
    print("  (assumes uniform sampling over canonical archs -- see note in section 5)")
    print(f"  {'N':>6}  {'P(collision)':>14}")
    print(f"  {'------'}  {'---------------'}")
    for n in [50, 100, 200, 500, 1000, 1200]:
        print(f"  {n:>6}  {p_collision_birthday(n)*100:>13.3f}%")

    print("\n-- 5. Multiplicity non-uniformity (op contribution only)")
    print("  Simple architectures are overrepresented in the raw encoding space.")
    print("  Op multiplicity lower bound = 3^(5-k) for k active intermediate nodes:")
    print()
    print(f"  {'k (active nodes)':>18}  {'Op multiplicity (>=)':>20}  {'Arch type'}")
    print(f"  {'------------------'}  {'--------------------'}  {'-----------'}")
    for k in range(1, N_OPS + 1):
        arch_type = "minimal bottleneck" if k == 1 else ("fully active" if k == N_OPS else "")
        print(f"  {k:>18}  {op_multiplicity(k):>20,}  {arch_type}")
    print()
    print("  Edge multiplicity (edges to dead nodes) adds further bias beyond these values.")
    print("  True collision probability for simple archs is higher than section 4 suggests.")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
