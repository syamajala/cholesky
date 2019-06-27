"""
Copyright 2019 Stanford University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import scipy
import scipy.linalg
import scipy.io
import numpy as np
import pandas as pd
import os
import collections


def eval_line(line, offset):
    line = line[offset:]
    line = eval(line)
    return line


def print_matrix(mat):
    r, c = mat.shape

    for i in range(0, r):
        row = list(map(lambda c: float("%0.2f" % c), mat[i]))
        print(row)


def potrf(mat, bounds):
    rA, cA = bounds["A"]
    mat[rA, cA] = scipy.linalg.cholesky(mat[rA, cA], lower=True)


def trsm(mat, bounds):
    rA, cA = bounds["A"]
    rB, cB = bounds["B"]
    mat[rB, cB] = scipy.linalg.solve(mat[rA, cA], mat[rB, cB].T).T


def gemm(mat, bounds):
    rA, cA = bounds["A"]
    rB, cB = bounds["B"]
    rC, cC = bounds["C"]
    mat[rC, cC] = mat[rC, cC] - mat[rA, cA].dot(mat[rB, cB].T)

    if bounds["A"] == bounds["B"]:
        mat[rC, cC] = np.tril(mat[rC, cC])


def compute_bounds(line):
    bounds = {}

    for block in ["A", "B", "C"]:
        lo = f"{block}_Lo"
        hi = f"{block}_Hi"
        if lo in line and hi in line:
            lo = line[lo]
            hi = line[hi]
            row = (lo[0], hi[0]+1)
            col = (lo[1], hi[1]+1)

            bounds[block] = (slice(*row), slice(*col))

    return bounds


def find_file(line):

    blocks = {}

    for block in ["A", "B", "C"]:
        if block in line:
            blocks[block] = '%d%d' % (line[block][0], line[block][1])

    if line['op'] == "POTRF":
        op = f"potrf_lvl{line['Level']}_a{blocks['A']}.mtx"
    elif line['op'] == "TRSM":
        op = f"trsm_lvl{line['Level']}_a{blocks['A']}_b{blocks['B']}.mtx"
    elif line['op'] == "GEMM":
        op = f"gemm_lvl{line['Level']}_a{blocks['A']}_b{blocks['B']}_c{blocks['C']}.mtx"

    return op


def verify(line, mat, bounds, directory=""):
    output_file = find_file(line)
    print("Verifying:", output_file)
    output_file = os.path.join(directory, output_file)
    output = scipy.io.mmread(output_file)
    output = output.toarray()
    output = np.tril(output)

    try:
        for idx, row in bounds.iterrows():
            lo = row["Lo"]
            hi = row["Hi"]

            rV = slice(lo[0], hi[0]+1)
            cV = slice(lo[1], hi[1]+1)

            assert np.allclose(mat[rV, cV], output[rV, cV], rtol=1e-04, atol=1e-04) is True
    except AssertionError as ex:
        print(f"{row['color']} differ!")
        diff = mat - output
        print("Python:")
        print_matrix(mat[rV, cV])
        print()
        print("Regent:")
        print_matrix(output[rV, cV])
        print()
        print("Diff:")
        print_matrix(diff[rV, cV])
        raise ex


def permute_matrix(matrix_file, separator_file):

    mat = scipy.io.mmread(matrix_file)
    mat = mat.toarray()

    pmat = np.zeros(mat.shape)

    separators = {}
    levels = 0
    num_separators = 0

    with open(separator_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                levels, num_separators = line.split(" ")
                levels = int(levels)
                num_separators = int(num_separators)
                continue

            sep, dofs = line.split(";")
            sep = int(sep)+1
            dofs = dofs.rstrip().split(",")

            if dofs[-1] == '':
                dofs = dofs[:-1]

            dofs = list(map(int, dofs))

            separators[sep] = dofs

    tree = []

    end = None
    start = 1
    for level in range(levels-1, -1, -1):
        if end is None:
            end = 2**level+1
        else:
            end = start + 2**level
        seps = list(range(start, end))
        tree.append(seps)
        start = end

    sep_bounds = {}
    i, j = 0, 0
    nzs = collections.defaultdict(lambda: 0)

    for level, seps in enumerate(tree):
        # print("Level:", level, "Separators:", seps)
        for sep in seps:
            sep_bounds[sep] = (i, j)

            dofs = separators[sep]
            # print("\tSeparator:", sep, "Dofs:", dofs)
            for idxi, row in enumerate(dofs):
                for idxj, col in enumerate(dofs):
                    if idxj <= idxi and mat[row, col]:
                        pmat[i+idxi, j+idxj] = mat[row, col]
                        # print("Filling:", sep, sep, i+idxi, j+idxj, "with I:", i, "J:", j, "Val:", mat[row, col])
                        nzs[(sep, sep)] += 1
            i += (idxi + 1)
            j += (idxj + 1)

    for level, seps in enumerate(tree):
        for sep_idx, sep in enumerate(seps):

            par_idx = sep_idx

            for par_level in range(level+1, levels):
                par_idx = int(par_idx/2)
                par_sep = tree[par_level][par_idx]

                row = sep_bounds[par_sep]
                col = sep_bounds[sep]

                lx, _ = row
                _, ly = col
                # print ("Sep:", sep, "Bounds:", col, "Par Sep:", par_sep, "Bounds:", row, "Start:", (lx, ly))

                for idxi, i in enumerate(separators[par_sep]):
                    for idxj, j in enumerate(separators[sep]):
                        if mat[i, j]:
                            # print("Filling:", par_sep, sep, lx+idxi, ly+idxj, "with I:", i, "J:", j, "Val:", mat[i, j])
                            nzs[(par_sep, sep)] += 1
                        pmat[lx+idxi, ly+idxj] = mat[i, j]

    return (nzs, pmat)


def debug_factor(matrix_file, separator_file, factored_mat, log_file, directory=""):
    nzs, mat = permute_matrix(matrix_file, separator_file)

    omat = np.array(mat)

    cholesky_numpy = scipy.linalg.cholesky(omat, lower=True)

    cholesky_regent = scipy.io.mmread(factored_mat)
    cholesky_regent = cholesky_regent.toarray()
    cholesky_regent = np.tril(cholesky_regent)

    last_block = None
    last_line = None
    op = None
    blocks = []
    clusters = []

    with open(log_file, 'r') as f:
        for line in f:
            line = line.lstrip().rstrip()
            if line.startswith("Block:"):
                line = eval_line(line, line.index(':')+1)
                blocks.append(line)
            elif line.startswith("Cluster:"):
                line = eval_line(line, line.index(':')+1)
                clusters.append(line)

    blocks = pd.DataFrame(blocks)
    clusters = pd.DataFrame(clusters)

    with open(log_file, 'r') as f:
        for line in f:
            line = line.lstrip().rstrip()
            if line.startswith("POTRF:"):
                operation = potrf
                op = "POTRF"
            elif line.startswith("TRSM:"):
                operation = trsm
                op = "TRSM"
            elif line.startswith("GEMM:"):
                operation = gemm
                op = "GEMM"
            else:
                continue

            line = eval_line(line, line.index(':')+1)
            line["op"] = op
            bounds = compute_bounds(line)
            operation(mat, bounds)

            if last_block is None:
                last_block = line["Block"]
            elif last_block != line["Block"] or last_line['op'] != op:
                part = clusters[(clusters['Interval'] == last_line['Interval']) & (clusters['Block'] == last_block)]
                verify(last_line, mat, part, directory)
                last_block = line["Block"]

            last_line = line

    print(np.allclose(cholesky_numpy, cholesky_regent, rtol=1e-04, atol=1e-04))


def check_matrix(matrix_file, separator_file, factored_mat):
    nzs, mat = permute_matrix(matrix_file, separator_file)
    cholesky_numpy = scipy.linalg.cholesky(mat, lower=True)

    cholesky_regent = scipy.io.mmread(factored_mat)
    cholesky_regent = cholesky_regent.toarray()
    cholesky_regent = np.tril(cholesky_regent)

    res = np.allclose(cholesky_numpy, cholesky_regent, rtol=1e-04, atol=1e-04)
    return res, cholesky_numpy, cholesky_regent


def check_solution(A, b, solution_regent):
    A = scipy.io.mmread(A)
    A = A.toarray()

    b = scipy.io.mmread(b)

    solution_regent = np.genfromtxt(solution_regent)
    solution_regent = solution_regent.reshape(b.shape)

    solution_numpy = scipy.linalg.solve(A, b)

    res = np.allclose(solution_numpy, solution_regent, rtol=1e-04, atol=1e-04)
    return res


def generate_b(n):
    np.random.seed()
    a = np.random.randint(1, 11, size=(n, 1))
    scipy.io.mmwrite("B_%dx1.mtx" % n, a)


debug_factor("tests/lapl_3375x3375/lapl_15_3.mtx", "tests/lapl_3375x3375/lapl_15_3_ord_5.txt", "factored_matrix.mtx", "output", "steps")
