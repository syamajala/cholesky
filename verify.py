import scipy
import scipy.linalg
import scipy.io
import numpy as np
import re
import os
import itertools as it


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


arg_map = {0: "A", 1: "B", 2: "C"}


def compute_bounds(line):
    los = re.findall(r'Lo: \d+ \d+', line)
    his = re.findall(r'Hi: \d+ \d+', line)

    los = list(map(lambda lo: lo.split(" "), los))
    los = list(map(lambda lo: (int(lo[1]), int(lo[2])), los))

    his = list(map(lambda hi: hi.split(" "), his))
    his = list(map(lambda hi: (int(hi[1])+1, int(hi[2])+1), his))

    bounds = {}
    for idx, lo_hi in enumerate(list(zip(los, his))):
        lo, hi = lo_hi
        row = (lo[0], hi[0])
        col = (lo[1], hi[1])
        bounds[arg_map[idx]] = (slice(*row), slice(*col))

    return bounds


def find_file(line, directory=""):

    lvl = re.findall(r'Level: \d+', line)[0]
    lvl = int(line.split(" ")[1])

    blocks = re.findall(r'\((.*?,.*?)\)', line)
    blocks = list(map(lambda b: b.split(','), blocks))
    blocks = list(map(lambda b: (int(b[0]), int(b[1])), blocks))
    blocks = list(it.chain.from_iterable(blocks))

    if "POTRF" in line:
        op = "potrf_lvl%d_a%d%d.mtx" % (lvl, *blocks)
    elif "TRSM" in line:
        op = "trsm_lvl%d_a%d%d_b%d%d.mtx" % (lvl, *blocks)
    elif "GEMM" in line:
        op = "gemm_lvl%d_a%d%d_b%d%d_c%d%d.mtx" % (lvl, *blocks)

    return os.path.join(directory, op)


def verify(line, mat, bounds, directory=""):
    print("Verifying:", line)
    output_file = find_file(line, directory)
    output = scipy.io.mmread(output_file)
    output = output.toarray()
    output = np.tril(output)

    try:
        assert(np.allclose(mat, output, rtol=1e-04, atol=1e-04))
    except AssertionError as ex:
        diff = mat - output
        print("Python:")
        print_matrix(mat)
        print()
        print("Regent:")
        print_matrix(output)
        print()
        print("Diff:")
        print_matrix(diff)
        raise ex


def debug(permuted_mat, factored_mat, output, directory=""):
    permuted_mat = os.path.join(directory, permuted_mat)
    factored_mat = os.path.join(directory, factored_mat)
    output = os.path.join(directory, output)

    mat = scipy.io.mmread(permuted_mat)
    mat = mat.toarray()
    mat = np.tril(mat)

    omat = scipy.io.mmread(permuted_mat)
    omat = omat.toarray()
    omat = np.tril(omat)

    cholesky_numpy = scipy.linalg.cholesky(omat, lower=True)

    cholesky_regent = scipy.io.mmread(factored_mat)
    cholesky_regent = cholesky_regent.toarray()
    cholesky_regent = np.tril(cholesky_regent)

    with open(output, 'r') as f:
        for line in f:
            line = line.lstrip().rstrip()
            if line.startswith("Level"):
                op_line = line
                if "POTRF" in line:
                    operation = potrf
                elif "TRSM" in line:
                    operation = trsm
                elif "GEMM" in line:
                    operation = gemm
                else:
                    operation = None
            elif line.startswith("Size"):
                if operation:
                    bounds = compute_bounds(line)
                    operation(mat, bounds)
                    verify(op_line, mat, bounds, directory)

    print(np.allclose(cholesky_numpy, cholesky_regent, rtol=1e-04, atol=1e-04))


def check_matrix(permuted_mat, factored_mat):
    mat = scipy.io.mmread(permuted_mat)
    mat = mat.toarray()
    mat = np.tril(mat)

    cholesky_numpy = scipy.linalg.cholesky(mat, lower=True)

    cholesky_regent = scipy.io.mmread(factored_mat)
    cholesky_regent = cholesky_regent.toarray()
    cholesky_regent = np.tril(cholesky_regent)

    res = np.allclose(cholesky_numpy, cholesky_regent, rtol=1e-04, atol=1e-04)
    return res


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
