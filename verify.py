import scipy
import scipy.linalg
import scipy.io
import numpy as np

a = scipy.io.mmread("permuted_matrix.mtx")
a = a.toarray()
cholesky_a = scipy.linalg.cholesky(a, lower=True)

factored_a = scipy.io.mmread("factored_matrix.mtx")
factored_a = factored_a.toarray()
factored_a = np.tril(factored_a)


def print_matrix(mat):
    r, c = mat.shape

    for i in range(0, r):
        row = list(map(lambda c: float("%0.2f" % c), mat[i]))
        print(row)


# print_matrix(cholesky_a)
# print()
# print_matrix(factored_a)

print(np.allclose(factored_a, cholesky_a))
