import copy
from typing import List, Optional

import numpy as np

# from application.function import Function
# from strategy.genetics.genetic_base import BaseProgram
from strategy.genetics.nasbench101_lib.model_spec import ModelSpec as _ModelSpec101

# Canonical op list for NASBench-101 hash_spec (must NOT include 'input'/'output')
_CANONICAL_OPS_101 = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']


class NASBENCH201:
    """
    CGP wrapper over a finite architecture set (e.g. NASBench-201).
    Structure is frozen; only function genes may mutate.
    """

    def __init__(self, op_list: List[str], random_state: np.random.RandomState, **kwargs):
        self.op_list = op_list
        self.random_state = random_state

        # super().__init__(**kwargs)
        self.op_to_idx = {op: i for i, op in enumerate(op_list)}
        self.idx_to_op = {i: op for i, op in enumerate(op_list)}

        self.nb_ops = self._sample_random_nb_ops()

        # Fixed CGP skeleton (10 nodes, 2-input)
        twoport = len(self.op_list) + 1  # + input
        self.genome = np.array([
            [0, 0, 0],  # 0 input
            [0, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 0],  # 3
            [twoport, 2, 3],  # 4
            [0, 0, 0],  # 5
            [0, 1, 0],  # 6
            [twoport, 5, 6],  # 7
            [0, 4, 0],  # 8
            [twoport, 7, 8],  # 9 output
        ], dtype=object)
        self._load_from_nb_ops()

    def _sample_random_nb_ops(self) -> np.ndarray:
        """
        Uniformly sample a NASBench-201 architecture.
        """
        return self.random_state.randint(
            low=0,
            high=len(self.op_list),
            size=6
        )

    def _load_from_nb_ops(self):
        """
        Load NASBench-201 genotype into a fixed CGP genome.
        """
        # Rows where NASBench ops live (canonical order)
        replace_rows = [1, 2, 3, 5, 6, 8]

        for gene_idx, row in enumerate(replace_rows):
            op_idx = self.nb_ops[gene_idx]
            self.genome[row, 0] = self.op_list[op_idx]

    def single_active_mutation(self, max_tries=100):
        for _ in range(max_tries):
            idx = self.random_state.randint(6)
            old = self.nb_ops[idx]

            new = old
            while new == old:
                new = self.random_state.randint(len(self.op_list))

            if new != old:
                self.nb_ops[idx] = new
                self._load_from_nb_ops()
                return self

        return self

    def point_mutation(self, p_point_replace):
        """
        CGP-consistent point mutation:
        Each NASBench edge (node analogue) is selected with probability p_point_replace.
        """
        n_edges = 6

        mutate_mask = (
                self.random_state.uniform(size=n_edges) <= p_point_replace
        )

        for i in range(n_edges):
            if not mutate_mask[i]:
                continue

            old = self.nb_ops[i]
            new = old
            while new == old:
                new = self.random_state.randint(len(self.op_list))

            self.nb_ops[i] = new

        self._load_from_nb_ops()
        return self

    def probabilistic_mutation(self, p_gene):
        """
        CGP-consistent probabilistic mutation:
        Each NASBench gene (edge operation) mutates independently with probability p_gene.
        """
        n_genes = 6

        for i in range(n_genes):
            if self.random_state.uniform() <= p_gene:
                old = self.nb_ops[i]
                new = old
                while new == old:
                    new = self.random_state.randint(len(self.op_list))
                self.nb_ops[i] = new

        self._load_from_nb_ops()
        return self

    def advance_random_state(self, seed):
        self.random_state.seed(seed)

    @property
    def arch_str(self) -> str:
        ops = [self.op_list[i] for i in self.nb_ops]

        return (
            f"|{ops[0]}~0|+"
            f"|{ops[1]}~0|{ops[2]}~1|+"
            f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
        )

    @property
    def tuple_program(self):
        """
        Canonical NASBench-201 genotype (6 categorical genes).
        """
        return tuple(int(x) for x in self.nb_ops)

    @property
    def tuple_program2 (self):
        """
        CGP NASBench-201 genotype (6 categorical genes).
        """
        active_genome = copy.deepcopy(self.genome)
        for node in active_genome:
            if isinstance(node[0], str):
                node[0] = self.op_to_idx[node[0]]
        return tuple(active_genome.reshape(-1))


    @property
    def nd_program(self):
        return np.asarray(self.nb_ops, dtype=np.int64)


class NASBENCH101:
    """
    Wrapper for NASBench-101 architectures.

    Encodes an architecture as a 26-dimensional integer vector:
      - genes [0:5]  → operation indices (0=conv3x3, 1=conv1x1, 2=maxpool3x3)
      - genes [5:26] → upper-triangular adjacency edges (binary 0/1)
    """

    # 3 intermediate ops (fixed)
    OP_LIST_DEFAULT = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
    N_OPS   = 5   # intermediate nodes (excl. input/output)
    N_EDGES = 21  # upper-triangular edges of a 7-node DAG

    def __init__(self, op_list: List[str], random_state: np.random.RandomState, **kwargs):
        self.op_list      = op_list          # e.g. ['conv3x3-bn-relu', ...]
        self.random_state = random_state
        self.nb_ops, self.nb_edges = self._sample_valid()
        self._arch_str_cache: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_random(self):
        ops = self.random_state.randint(0, len(self.op_list), size=self.N_OPS)
        # NASBench-101 search space: at most 9 edges in the original matrix
        n_edges = int(self.random_state.randint(1, 10))   # 1..9
        chosen  = self.random_state.choice(self.N_EDGES, size=n_edges, replace=False)
        edges   = np.zeros(self.N_EDGES, dtype=np.int64)
        edges[chosen] = 1
        return ops.astype(np.int64), edges

    def _sample_valid(self, max_tries: int = 1000):
        for _ in range(max_tries):
            ops, edges = self._sample_random()
            if self._is_valid(ops, edges):
                return ops, edges
            
        # Fallback: guaranteed minimal valid arch (input→output direct edge)
        ops   = np.zeros(self.N_OPS,   dtype=np.int64)
        edges = np.zeros(self.N_EDGES, dtype=np.int64)
        edges[self.N_EDGES - 1] = 1  # input(0) → output(6) is position (0,6) in upper-tri
        return ops, edges

    def _to_matrix(self, edges: np.ndarray) -> np.ndarray:
        """Reconstruct 7×7 upper-triangular adjacency matrix from 21-element edge vector."""
        mat = np.zeros((7, 7), dtype=int)
        k = 0
        for i in range(7):
            for j in range(i + 1, 7):
                mat[i, j] = int(edges[k])
                k += 1
        return mat

    def _to_ops(self, ops: np.ndarray) -> List[str]:
        return ['input'] + [self.op_list[int(o)] for o in ops] + ['output']

    def _spec(self, ops=None, edges=None) -> _ModelSpec101:
        ops   = self.nb_ops   if ops   is None else ops
        edges = self.nb_edges if edges is None else edges
        return _ModelSpec101(matrix=self._to_matrix(edges), ops=self._to_ops(ops))

    def _is_valid(self, ops=None, edges=None) -> bool:
        if edges is None:
            edges = self.nb_edges

        # NASBench-101 constraint: original adjacency matrix must have ≤9 edges
        if int(np.sum(edges)) > 9:
            return False
        return self._spec(ops, edges).valid_spec

    # ------------------------------------------------------------------
    # Mutation operators  (all retry up to max_tries=26; fall back to parent)
    # ------------------------------------------------------------------

    def single_active_mutation(self, max_tries: int = 26):
        """Flip one random gene, retry until valid."""
        ops_orig   = self.nb_ops.copy()
        edges_orig = self.nb_edges.copy()
        for _ in range(max_tries):
            idx = int(self.random_state.randint(self.N_OPS + self.N_EDGES))
            ops_new   = ops_orig.copy()
            edges_new = edges_orig.copy()
            if idx < self.N_OPS:
                old = ops_new[idx]
                new = old
                while new == old:
                    new = int(self.random_state.randint(len(self.op_list)))
                ops_new[idx] = new
            else:
                e = idx - self.N_OPS
                edges_new[e] = 1 - edges_new[e]  # toggle bit
            if self._is_valid(ops_new, edges_new):
                self.nb_ops, self.nb_edges = ops_new, edges_new
                self._arch_str_cache = None
                return self
        return self

    def point_mutation(self, p_point_replace: float):
        """
        CGP-pointwise mutation: a single mutate_mask is drawn for all
        26 genes and each selected gene is mutated once.  Falls back to parent
        if the result is architecturally invalid.
        """
        n_genes = self.N_OPS + self.N_EDGES
        mutate_mask = self.random_state.uniform(size=n_genes) <= p_point_replace

        ops_new   = self.nb_ops.copy()
        edges_new = self.nb_edges.copy()

        for i in range(self.N_OPS):
            if mutate_mask[i]:
                old = ops_new[i]
                new = old
                while new == old:
                    new = int(self.random_state.randint(len(self.op_list)))
                ops_new[i] = new

        for e in range(self.N_EDGES):
            if mutate_mask[self.N_OPS + e]:
                edges_new[e] = 1 - edges_new[e]

        if self._is_valid(ops_new, edges_new):
            self.nb_ops, self.nb_edges = ops_new, edges_new
            self._arch_str_cache = None
        return self

    def probabilistic_mutation(self, p_gene: float):
        """
        Each of the 26 genes mutates independently with probability p_gene.
        Falls back to parent if the result is invalid.
        """
        ops_new   = self.nb_ops.copy()
        edges_new = self.nb_edges.copy()

        for i in range(self.N_OPS):
            if self.random_state.uniform() <= p_gene:
                old = ops_new[i]
                new = old
                while new == old:
                    new = int(self.random_state.randint(len(self.op_list)))
                ops_new[i] = new

        for e in range(self.N_EDGES):
            if self.random_state.uniform() <= p_gene:
                edges_new[e] = 1 - edges_new[e]

        if self._is_valid(ops_new, edges_new):
            self.nb_ops, self.nb_edges = ops_new, edges_new
            self._arch_str_cache = None
        return self

    def advance_random_state(self, seed: int):
        self.random_state.seed(seed)

    # ------------------------------------------------------------------
    # Properties consumed by the SAMOS / duplicate-elimination pipeline
    # ------------------------------------------------------------------

    @property
    def arch_str(self) -> str:
        """
        MD5 hash of this architecture's isomorphism class — used as the
        lookup key into data_nasbench101.pkl.  Cached per instance; the
        cache is invalidated whenever a mutation changes nb_ops / nb_edges.
        Compatible with old pickled instances that pre-date the cache field.
        """
        if getattr(self, '_arch_str_cache', None) is None:
            spec = self._spec()
            if not spec.valid_spec:
                raise ValueError('arch_str requested on an invalid NASBench-101 spec')
            self._arch_str_cache = spec.hash_spec(_CANONICAL_OPS_101)
        return self._arch_str_cache

    @property
    def tuple_program(self):
        """26-element integer tuple; key for GPDuplicateElimination and surrogate fitting."""
        return tuple(int(x) for x in np.concatenate([self.nb_ops, self.nb_edges]))

    @property
    def nd_program(self) -> np.ndarray:
        return np.concatenate([self.nb_ops, self.nb_edges]).astype(np.int64)