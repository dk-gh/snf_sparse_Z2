import numpy as np
from itertools import product

from sparse_binary_matrix import SparseBinaryMatrix


def swap_rows(i0, i1, A):
    A[[i0, i1], :] = A[[i1, i0], :]


def swap_columns(j0, j1, A):
    A[:, [j0, j1]] = A[:, [j1, j0]]


def dense_smith_normal_form(A):
    """
    Input A is a numpy array with dtype=int
    and A[i,j] = 0 or 1 for all i,j

    Transforms A inplace into its smith normal form
    """
    number_of_rows, number_of_columns = A.shape
    n = 0

    while n < min(number_of_columns, number_of_rows):
        # identify the next non-zero entry and move this to
        # position A[n,n]
        for i, j in product(
            range(n, number_of_rows),
            range(n, number_of_columns)
        ):

            if A[i, j] == 1:
                swap_rows(n, i, A)
                swap_columns(n, j, A)
                break

        # perform row reductions
        for i in range(n+1, number_of_rows):
            if A[i, n] == 1:
                # perform bitwise xor element-wise on rows
                A[i] = A[i] ^ A[n]

        # perform column reductions
        for j in range(n+1, number_of_columns):
            if A[n, j] == 1:
                # perform bitwise xor element-wise on columns
                A[:, [j]] = A[:, [j]] ^ A[:, [n]]
        n = n+1

    return A


def check_random_sparse_matrix():
    n = 1000
    m = 2000

    # create a random n*m numpy array
    nums = np.ones(n*m, dtype=int)
    nums[:int(0.999*n*m)] = 0
    np.random.shuffle(nums)
    nums = nums.reshape(n, m)

    # convert the numpy array to sparse format
    scn_sparse = SparseBinaryMatrix()
    scn_sparse.from_numpy_array(nums, validate=True)

    # transform to smith normal form
    scn_sparse.smith_normal_form()

    # validate sparse format
    scn_sparse.validate_synchronisation()

    # convert from sparse back to numpy array
    scn_out = scn_sparse.to_numpy_array(validate=True)

    snf = dense_smith_normal_form(nums)

    assert(snf == scn_out).all()

    assert (scn_sparse.trace() == scn_out.sum())


if __name__ == '__main__':

    check_random_sparse_matrix()
