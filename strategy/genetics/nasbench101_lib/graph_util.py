# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions used by generate_graph.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import itertools

import numpy as np


def gen_is_edge_fn(bits):
  def is_edge(x, y):
    if x >= y:
      return 0
    index = x + (y * (y - 1) // 2)
    return (bits >> index) % 2 == 1
  return np.vectorize(is_edge)


def is_full_dag(matrix):
  shape = np.shape(matrix)
  rows = matrix[:shape[0]-1, :] == 0
  rows = np.all(rows, axis=1)
  rows_bad = np.any(rows)
  cols = matrix[:, 1:] == 0
  cols = np.all(cols, axis=0)
  cols_bad = np.any(cols)
  return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
  return np.sum(matrix)


def hash_module(matrix, labeling):
  vertices = np.shape(matrix)[0]
  in_edges = np.sum(matrix, axis=0).tolist()
  out_edges = np.sum(matrix, axis=1).tolist()
  assert len(in_edges) == len(out_edges) == len(labeling)
  hashes = list(zip(out_edges, in_edges, labeling))
  hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
  for _ in range(vertices):
    new_hashes = []
    for v in range(vertices):
      in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
      out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
      new_hashes.append(hashlib.md5(
          (''.join(sorted(in_neighbors)) + '|' +
           ''.join(sorted(out_neighbors)) + '|' +
           hashes[v]).encode('utf-8')).hexdigest())
    hashes = new_hashes
  return hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()


def permute_graph(graph, label, permutation):
  forward_perm = zip(permutation, list(range(len(permutation))))
  inverse_perm = [x[1] for x in sorted(forward_perm)]
  edge_fn = lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1
  new_matrix = np.fromfunction(np.vectorize(edge_fn),
                               (len(label), len(label)),
                               dtype=np.int8)
  new_label = [label[inverse_perm[i]] for i in range(len(label))]
  return new_matrix, new_label


def is_isomorphic(graph1, graph2):
  matrix1, label1 = np.array(graph1[0]), graph1[1]
  matrix2, label2 = np.array(graph2[0]), graph2[1]
  assert np.shape(matrix1) == np.shape(matrix2)
  assert len(label1) == len(label2)
  vertices = np.shape(matrix1)[0]
  for perm in itertools.permutations(range(0, vertices)):
    pmatrix1, plabel1 = permute_graph(matrix1, label1, perm)
    if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
      return True
  return False
