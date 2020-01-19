""" M4N9 Computational Linear Algebra - Project 2
Tudor Trita Trita
CID: 01199397

Implemented using Python 3.7 (need f-string python 3.6+)

Question 2. LU Factorisation of Banded Matrices & some tests for
            possible cases of a matrix input
"""
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def bandedLU(M, ml, mu, output='sparse'):
    """LU Decomposition for a banded matrix M with lower band ml, upper
       band mu. We can specify the output the function to be either in
       csr_matrix format or numpy dense format.
    """
    assert M.shape[0] == M.shape[1], "Matrix is not square"
    assert len(M.shape) == 2, "Input a 2D array"
    N = M.shape[0]
    dtype = M[0, 0].dtype
    assert dtype in [np.int, np.float, np.complex], "Data type not understood"

    # Cast matrix to double precision, (ints get cast into float64's):
    M = M.astype(np.complex128) if dtype==np.complex else M.astype(np.float64)
    U = scipy.sparse.csr_matrix(M)  # Sparse copy of M, in case M isn't sparse

    # Preallocating elements of L which will become (otherwise inefficiency warning)
    L_ones = scipy.sparse.diags([1]*(ml+1), -np.arange(ml+1), shape=(N, N), dtype=dtype)
    L = L_ones.tocsr()

    # We implement the algorithm vectorising across the j's
    for k in range(N - 1):
        n = min(k+mu+1, N)
        for j in range(k + 1, min(k+ml+1, N)):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:n] -= L[j, k]*U[k, k:n]

    if output == 'sparse':
        return L, U

    elif output == 'dense':
        return L.toarray(), U.toarray()

    else:
        raise Exception('Enter a valid output')


def tests(M, L, U, ml, mu, calc_dets=False, display=False):
    """Implements various tests as described in the report"""
    print(f"\n TESTS FOR MATRIX OF DIMENSIONS {M.shape} BANDS ml={ml}, mu={mu}\n")
    if display:
        print("MATRIX M: \n\n")
        np.set_printoptions(2)
        print(M.todense())

    # Computing different norms, determinants, etc.
    sparse_bool = scipy.sparse.issparse(M)
    if sparse_bool:
        NORM_M = scipy.sparse.linalg.norm(M)
        NORM_DIFF = scipy.sparse.linalg.norm(M - L @ U)
    else:
        NORM_M = np.linalg.norm(M)
        NORM_DIFF = np.linalg.norm(M - L @ U)

    threshold_error = 10**-10
    threshold_norm = 10**-10
    threshold_dets = 10**-8
    threshold_zeros = 10**-10

    if calc_dets:
        if sparse_bool:
            DETM = np.linalg.det(M.todense())
            DETU = np.linalg.det(U.todense())
            DETLU = np.linalg.det((L@U).todense())
            DETL = np.linalg.det(L.todense())
        else:
            DETM = np.linalg.det(M)
            DETU = np.linalg.det(U)
            DETLU = np.linalg.det(L@U)
            DETL = np.linalg.det(L)
        DETL_TIMES_U = DETL*DETU

        # Want to check all of these are 'close'
        DETS = np.array([DETM, DETLU, DETL_TIMES_U, DETU])
        DETS -= np.mean(DETS)
        # Checking all determinants are 'close' and that det_l is close to 1
        DETS_BOOL = bool((np.max(np.abs(DETS)) < threshold_dets)
                     and (np.abs(DETL - 1) < threshold_zeros))

    # TEST 1: Error of entries of LU
    E = M - L@U  # Error matrix
    largest_error, sum_errors = np.max(np.abs(E)), np.abs(E).sum()
    print("\n \t \t TEST 1:  \n \t    Difference M-LU: \n")
    print(f" ## |Largest Error| = {largest_error:.2E} ## ")
    print(f" ## Sum of |Errors| = {sum_errors:.2E} ##")
    print(f" ## |Largest Error| < T = {threshold_error:.1E}? ##")
    print(f"\t \t {bool(largest_error < threshold_error)} \n")

    # TEST 2: Check norms
    print("\n\t\tTEST 2:\n\t     Various Norms:\n")
    print(f" ## \t ||M|| = {NORM_M:.2E} ## ")
    print(f" ## ||M - LU|| = {NORM_DIFF:.2E} ##")
    print(f" ## ||M - LU|| < T = {threshold_norm:.1E}? ##")
    print(f"\t\t{bool(NORM_DIFF < threshold_norm)} \n")

    # Test 3: Check that U is an upper-banded matrix with band mu
    largest_error, sum_errors = 0, 0
    # Sum of errors across lower bands of U (avoids turning into dense)
    if ml and mu:
        sum_errors = [(sum_errors + np.sum(np.abs(U.diagonal(i)))) for i in (-np.arange(1, ml+1))][0]
        largest_error = np.max([np.max(np.abs(U.diagonal(i))) for i in (-np.arange(1, ml+1))])
    else:
        sum_errors, largest_error = np.nan, np.nan
    # Max of errors across lower bands of U (avoids turning into dense)
    print("\n \t \t TEST 3:\n\t      U Upper Triangular?:\n")
    print(f" ## |Largest Error| = {largest_error:.2E} ## ")
    print(f" ## Sum of |Errors| = {sum_errors:.2E} ##")
    print(f" ## |Largest Error| < T = {threshold_error:.1E}? ##")
    print(f"\t \t {bool(largest_error < threshold_error)} \n")

    if calc_dets:
        # Test 4: Check determinants det(M) = det(LU) = det(L)det(U) = det(U)
        # and det(L) close to 1
        print("\n \t \t TEST 4:\n\t      Determinants:\n")
        print(f" ## det(M) = {DETM:.2E} ##")
        print(f" ## det(LU) = {DETLU:.2E} ##")
        print(f" ## det(L)det(U) = {DETL_TIMES_U:.2E} ##")
        print(f" ## det(U) = {DETU:.2E} ##")
        print(f" ## det(L) = {DETL:.2E} ##")
        print(f" All values close and det(L) ~ 1?")
        print(f"\t\t{DETS_BOOL} \n")
    return


def create_banded_matrix(N, ml, mu, type, M=None, c=1, output='sparse'):
    if not M:
        if type=='real':
            M = np.random.randn(N, N)*c
            M = np.triu(M, -ml) - np.triu(M, mu+1)
        elif type=='complex':
            M1 = np.random.randn(N, N)*c
            M2 = np.random.randn(N, N)*c
            M1 = np.triu(M1, -ml) - np.triu(M1, mu+1)
            M2 = np.triu(M2, -ml) - np.triu(M2, mu+1)
            M = M1 + 1j*M2

    if output=='sparse':
        M = scipy.sparse.csr_matrix(M)
        return M
    elif output=='dense':
        return M


def main():
    """Short summary.
    """
    # TEST SET ON REAL MATRICES

    # TEST 1.1: RANDOM REAL MATRIX EQUAL BANDS
    N, ml, mu = 7, 4, 4
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 1.2: RANDOM REAL MATRIX NON-EQUAL BANDS
    N, ml, mu = 7, 5, 2
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 1.3: RANDOM REAL MATRIX UPPER-TRIANGULAR
    N, ml, mu = 7, 0, 6
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 1.4: RANDOM REAL MATRIX DIAGONAL
    N, ml, mu = 7, 0, 0
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 1.5: RANDOM REAL MATRIX LARGE
    N, ml, mu = 100, 10, 30
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=False, display=True)

    # TEST 1.5: MATRIX WITH LARGE ENTRIES
    N, ml, mu = 10, 2, 2
    M = create_banded_matrix(N, ml, mu, 'real', c=pow(10, 4), output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=False, display=True)

    # TEST 1.6: MATRIX WITH TINY ENTRIES
    N, ml, mu = 10, 2, 2
    M = create_banded_matrix(N, ml, mu, 'real', c=pow(10, -6), output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=False, display=True)

    # TEST 1.6: MATRIX WITH FULL BANDS (NON-BANDED)
    N, ml, mu = 10, 9, 9
    M = create_banded_matrix(N, ml, mu, 'real', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=False, display=True)

    # TEST SET ON RANDOM COMPLEX MATRICES

    # TEST 2.1: RANDOM COMPLEX MATRIX EQUAL BANDS
    N, ml, mu = 7, 4, 4
    M = create_banded_matrix(N, ml, mu, 'complex', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 2.2: RANDOM REAL MATRIX NON-EQUAL BANDS
    N, ml, mu = 7, 5, 2
    M = create_banded_matrix(N, ml, mu, 'complex', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=True, display=True)

    # TEST 2.3: RANDOM COMPLEX MATRIX LARGE
    N, ml, mu = 100, 20, 10
    M = create_banded_matrix(N, ml, mu, 'complex', output='sparse')
    L, U = bandedLU(M, ml, mu)
    tests(M, L, U, ml, mu, calc_dets=False, display=True)
    return


if __name__ == "__main__":
    print("Program Started")
    print()
    main()
    print()
    print("Program Finished")
